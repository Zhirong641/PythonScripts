#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite-backed Image Tag Search (for 1M+ images)
Run:
  python app_sqlite.py --db /path/tags.db --host 0.0.0.0 --port 8000
"""

import argparse, math, mimetypes, os, sqlite3
from flask import Flask, request, render_template, send_file, abort, url_for, jsonify

GROUP_ORDER = ["rating", "general", "character", "meta", "year", "artist"]
GROUP_ORDER_CASE = (
    "CASE grp "
    + " ".join(f"WHEN '{grp}' THEN {idx}" for idx, grp in enumerate(GROUP_ORDER))
    + f" ELSE {len(GROUP_ORDER)} END"
)

def parse_query(q):
    if not q: return []
    out = []
    for raw in q.replace(",", " ").split():
        tok = raw.strip()
        if not tok: continue
        if ":" in tok:
            a,b = tok.split(":",1)
            try: thr = float(b.strip())
            except: thr = 0.5
            out.append((a.strip(), thr))
        else:
            out.append((tok, 0.5))
    return out

def compute_common_base(con):
    cur = con.cursor()
    cur.execute("SELECT path FROM images LIMIT 1000")  # sample
    paths = [r[0] for r in cur.fetchall()]
    if not paths: return None
    try:
        import os.path as op
        cb = op.commonpath(paths)
        if not os.path.isdir(cb):
            cb = op.dirname(cb)
        return cb
    except Exception:
        return os.path.dirname(paths[0])

def create_app(db_path):
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    import os
    common_base = os.environ.get("IMG_ROOT") or compute_common_base(con)
    print("Image root =", common_base, flush=True)

    app = Flask(__name__)

    @app.route("/")
    def home():
        q = request.args.get("q","").strip()
        page = max(int(request.args.get("page",1) or 1), 1)
        per_page = int(request.args.get("per_page", 60) or 60)

        results = []
        total = 0

        terms = parse_query(q)
        if terms:
            # Build dynamic SQL with N joins (AND semantics)
            # SELECT i.id,i.path, (it1.score + it2.score + ...) AS agg
            # FROM images i
            # JOIN image_tags it1 ... JOIN tags t1 ...
            # JOIN image_tags it2 ... JOIN tags t2 ...
            # ORDER BY agg DESC LIMIT ? OFFSET ?

            base = "SELECT i.id, i.path, {agg} AS agg FROM images i "
            joins = []
            agg_parts = []
            params = []
            alias_idx = 1
            for tag, thr in terms:
                it = f"it{alias_idx}"
                tg = f"tg{alias_idx}"
                joins.append(f"JOIN image_tags {it} ON {it}.image_id = i.id "
                             f"JOIN tags {tg} ON {tg}.id = {it}.tag_id AND {tg}.name = ? AND {it}.score >= ?")
                params.extend([tag, thr])
                agg_parts.append(f"{it}.score")
                alias_idx += 1
            sql = base.format(agg=" + ".join(agg_parts)) + " ".join(joins)
            # Get total count via subquery
            count_sql = "SELECT COUNT(*) FROM (" + sql + ")"
            cur = con.cursor()
            cur.execute(count_sql, params)
            total = cur.fetchone()[0]

            sql += " ORDER BY agg DESC, i.path LIMIT ? OFFSET ?"
            params2 = params + [per_page, (page-1)*per_page]
            cur.execute(sql, params2)
            rows = cur.fetchall()
            for r in rows:
                # tooltip: fetch top 30 tags for this image quickly
                tt_rows = con.execute(
                    "SELECT tags.name, COALESCE(tags.ja,'') ja, image_tags.score, image_tags.grp "
                    "FROM image_tags JOIN tags ON tags.id=image_tags.tag_id "
                    "WHERE image_tags.image_id=? ORDER BY image_tags.score DESC LIMIT 30", (r["id"],)
                ).fetchall()
                tooltip = "\n".join([
                    f"[{x['grp']}] {x['name']}"
                    + (f" ({x['ja']})" if x['ja'] else "")
                    + f" {x['score']:.3f}"
                    for x in tt_rows
                ])
                results.append({
                    "path": r["path"],
                    "thumb_url": url_for("serve_image", path=r["path"]),
                    "detail_url": url_for("detail", image_id=r["id"]),
                    "tooltip": tooltip,
                    "agg": float(r["agg"]),
                })

        total_pages = (total + per_page - 1)//per_page if per_page else 1
        return render_template("index.html",
                               q=q, results=results, page=page, per_page=per_page,
                               total=total, total_pages=total_pages)

    @app.route("/detail")
    def detail():
        image_id = int(request.args.get("image_id", "0") or 0)
        r = con.execute("SELECT id, path FROM images WHERE id=?", (image_id,)).fetchone()
        if not r: abort(404)
        rows = con.execute(
            "SELECT grp as group_name, tags.name as tag, COALESCE(tags.ja,'') as ja, image_tags.score as score "
            "FROM image_tags JOIN tags ON tags.id=image_tags.tag_id "
            "WHERE image_tags.image_id=? ORDER BY "
            f"{GROUP_ORDER_CASE}, score DESC",
            (image_id,),
        ).fetchall()
        mapped = [{"group": x["group_name"], "tag": x["tag"], "ja": x["ja"], "score": x["score"]} for x in rows]
        return render_template("detail.html",
                               path=r["path"],
                               img_url=url_for("serve_image", path=r["path"]),
                               rows=mapped)

    @app.route("/img")
    def serve_image():
        path = request.args.get("path","")
        if not path: abort(404)
        if common_base and not os.path.realpath(path).startswith(os.path.realpath(common_base)):
            abort(403)
        if not os.path.exists(path): abort(404)
        mt,_ = mimetypes.guess_type(path)
        return send_file(path, mimetype=mt or "application/octet-stream")

    @app.get("/api/suggest")
    def api_suggest():
        prefix = (request.args.get("q") or "").strip().lower()
        limit  = int(request.args.get("limit", 30) or 30)
        if not prefix:
            return jsonify([])

        # 模糊匹配（大小写不敏感），支持英文 name 和日文 ja
        # pos: 命中位置，越靠前越好；再按 freq 降序
        pattern = f"%{prefix}%"
        rows = con.execute(
            """
            SELECT
            name,
            COALESCE(ja,'') AS ja,
            COALESCE(freq,0) AS freq,
            MIN(NULLIF(INSTR(LOWER(name), ?), 0), NULLIF(INSTR(LOWER(COALESCE(ja,'')), ?), 0)) AS pos
            FROM tags
            WHERE LOWER(name) LIKE ? OR LOWER(ja) LIKE ?
            GROUP BY name
            ORDER BY
            COALESCE(pos, 999999) ASC,
            freq DESC,
            name ASC
            LIMIT ?
            """,
            (prefix, prefix, pattern, pattern, limit)
        ).fetchall()

        return jsonify([{"tag": r["name"], "ja": r["ja"], "freq": r["freq"]} for r in rows])

    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = create_app(args.db)
    try:
        from waitress import serve
        print(f"Serving with waitress on http://{args.host}:{args.port}")
        serve(app, host=args.host, port=args.port)
    except Exception:
        app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
