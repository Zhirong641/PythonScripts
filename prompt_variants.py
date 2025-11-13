# -*- coding: utf-8 -*-
import random, re
from typing import List, Optional

# ===== 可选：OpenCLIP 分词，贴近 SDXL 77 token 上限 =====
try:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    def _tok_len(s: str) -> int:
        return len(_tok(s, add_special_tokens=False)["input_ids"])
except Exception:
    _tok = None
    def _tok_len(s: str) -> int:
        # 简易估算（略保守）：英文词/短语≈1 token，汉字按字计
        words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s\w]", s.replace("_", " "))
        return len(words)

_COLOR_TOKENS = {
    "black", "white", "gray", "grey", "silver", "gold", "golden",
    "brown", "chestnut", "auburn", "hazel", "amber",
    "blonde", "platinum", "ginger", "copper", "bronze", "strawberry", "honey",
    "red", "crimson", "scarlet", "maroon", "burgundy",
    "pink", "magenta", "rose", "fuchsia",
    "orange", "peach", "apricot",
    "yellow",
    "green", "emerald", "jade", "mint", "lime", "olive",
    "teal", "turquoise", "aqua", "aquamarine", "cyan",
    "blue", "azure", "indigo", "navy", "cerulean", "sapphire",
    "purple", "violet", "lavender", "lilac",
}

_COLOR_SPECIAL = {
    "light brown", "dark brown", "light blue", "dark blue", "light green", "dark green",
    "light pink", "dark pink", "light purple", "dark purple", "light gray", "dark gray",
    "ash blonde", "dirty blonde", "platinum blonde", "strawberry blonde",
    "sky blue", "navy blue", "royal blue", "sea green",
    "two tone", "two-tone", "gradient", "multicolored", "multi colored", "rainbow", "pastel rainbow",
}

_COLOR_CONNECTORS = {
    "and", "with", "of", "to", "from", "the", "in", "on", "at", "for",
    "tips", "tipped", "streaks", "streak", "streaked", "ends", "inner", "outer",
    "colored", "colour", "highlight", "highlights", "highlighted",
    "two", "tone", "tones", "dual", "split", "gradient", "ombre",
}

_COLOR_MODIFIERS = {
    "light", "dark", "very", "pale", "pastel", "bright", "deep", "soft",
    "vivid", "neon", "warm", "cool", "muted", "faint", "bold", "rich",
    "ash", "dirty", "dull", "glossy", "frosted",
}


def _is_color_tag(tag: str, suffix: str) -> bool:
    suffix = suffix.strip()
    if not suffix:
        return False
    if not tag.endswith(f" {suffix}"):
        return False
    base = tag[: -(len(suffix) + 1)].strip()
    if not base:
        return False

    base_norm = base.replace("-", " ")
    if base_norm in _COLOR_SPECIAL:
        return True

    tokens = [tok for tok in re.split(r"[\s/-]+", base_norm) if tok]
    filtered = [
        tok for tok in tokens
        if tok not in _COLOR_CONNECTORS and tok not in _COLOR_MODIFIERS
    ]
    if not filtered:
        return False
    return all(tok in _COLOR_TOKENS for tok in filtered)


_HAIR_LENGTH_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"absurdly\s+long", r"very\s+long", r"waist(?:\s+|-)length", r"hip(?:\s+|-)length",
        r"thigh(?:\s+|-)length", r"knee(?:\s+|-)length", r"ankle(?:\s+|-)length",
        r"floor(?:\s+|-)length", r"shoulder(?:\s+|-)length", r"medium(?:\s+|-)length",
        r"neck(?:\s+|-)length", r"chin(?:\s+|-)length", r"very\s+short", r"short",
        r"medium", r"long",
    ]) +
    r")\s+hair\b"
)

_HAIRSTYLE_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"twin(?:\s+)?tails?", r"twintails", r"twin braids?", r"double braids?", r"single braid",
        r"side braid", r"french braid", r"loose braid", r"braids?", r"ponytail", r"high ponytail",
        r"low ponytail", r"side ponytail", r"twin ponytail", r"twin buns?", r"double buns?",
        r"buns?", r"odango", r"pigtails?", r"drill hair", r"ringlets", r"ahoge", r"spiky hair",
        r"curly hair", r"wavy hair", r"straight hair", r"mohawk", r"buzz cut", r"pixie cut",
        r"bob cut",
    ]) +
    r")\b"
)

_BUST_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"flat chest", r"small breasts?", r"medium breasts?", r"large breasts?",
        r"huge breasts?", r"gigantic breasts?", r"massive breasts?", r"big breasts?",
    ]) +
    r")\b"
)

_SLEEVE_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"sleeveless(?:\s+dress)?", r"strapless", r"short sleeves?", r"short-sleeved",
        r"long sleeves?", r"long-sleeved", r"detached sleeves?", r"single sleeve",
        r"no sleeves", r"bare shoulders", r"off-shoulder", r"one-shoulder",
        r"rolled(?:\s+up)? sleeves?", r"puffy sleeves?", r"bell sleeves?",
    ]) +
    r")\b"
)

_DRESS_SKIRT_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"sundress", r"one-piece dress", r"short dress", r"mini dress", r"long dress",
        r"evening dress", r"gown", r"ball gown", r"dress", r"mini skirt", r"micro skirt",
        r"pleated skirt", r"long skirt", r"skirt",
    ]) +
    r")\b"
)

_LEGWEAR_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"thighhighs?", r"thigh highs?", r"overknee socks?", r"over-the-knee socks?",
        r"knee[-\s]?highs?", r"kneesocks?", r"knee socks?", r"crew socks?", r"ankle socks?",
        r"loose socks?", r"socks?", r"stockings?", r"fishnet stockings?", r"fishnets?",
        r"pantyhose", r"tights", r"leggings", r"leg warmers?", r"garter belts?",
        r"garter straps?", r"bodystocking", r"bare legs?",
    ]) +
    r")\b"
)

_FOOTWEAR_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"barefoot", r"boots?", r"ankle boots?", r"knee boots?", r"thigh boots?", r"lace-up boots?",
        r"combat boots?", r"loafers?", r"sneakers?", r"running shoes?", r"athletic shoes?",
        r"dress shoes?", r"platform shoes?", r"heels?", r"high heels?", r"pumps?", r"mary janes?",
        r"sandals?", r"geta", r"zori", r"okobo", r"slippers?",
    ]) +
    r")\b"
)

_RIBBON_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"hair ribbon", r"hair bow", r"bowtie", r"bow", r"ribbon",
    ]) +
    r")\b"
)

_BAG_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"handbag", r"shoulder bag", r"messenger bag", r"tote bag", r"bag", r"backpack",
        r"satchel", r"purse", r"fanny pack", r"waist bag", r"hip pack", r"briefcase",
    ]) +
    r")\b"
)

_GLASSES_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"glasses", r"eyeglasses", r"spectacles", r"sunglasses", r"eyewear", r"goggles", r"visor",
        r"monocle",
    ]) +
    r")\b"
)

_HEADWEAR_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"hat", r"beret", r"cap", r"headband", r"hairband", r"hood", r"hooded", r"helmet",
        r"crown", r"tiara", r"veil", r"headdress",
    ]) +
    r")\b"
)

_BODY_VIEW_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"full body", r"upper body", r"upper torso", r"half body", r"cowboy shot",
        r"bust shot", r"bust", r"portrait", r"waist up", r"close[-\s]?up",
    ]) +
    r")\b"
)

_CAMERA_ANGLE_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"from above", r"from below", r"bird's eye", r"worm's eye", r"low angle",
        r"high angle", r"side view", r"profile view", r"front view", r"back view",
        r"rear view", r"three quarter view", r"three-quarter view", r"dutch angle",
    ]) +
    r")\b"
)

_POSE_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"standing", r"sitting", r"kneeling", r"crouching", r"squatting",
        r"lying", r"reclining", r"leaning",
    ]) +
    r")\b"
)

_PERSON_COUNT_F_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"[0-9]+(?:\+)?(?:\s*|[-_])*(?:girls?|women)",
        r"multi(?:ple)?(?:\s*|[-_])*(?:girls?|women)",
        r"double(?:\s*|[-_])*girls?", r"triple(?:\s*|[-_])*girls?", r"quadruple(?:\s*|[-_])*girls?",
    ]) +
    r")\b"
)

_PERSON_COUNT_M_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"[0-9]+(?:\+)?(?:\s*|[-_])*(?:boys?|men)",
        r"multi(?:ple)?(?:\s*|[-_])*(?:boys?|men)",
        r"double(?:\s*|[-_])*boys?", r"triple(?:\s*|[-_])*boys?", r"quadruple(?:\s*|[-_])*boys?",
    ]) +
    r")\b"
)

_PERSON_COUNT_GENERIC_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"[0-9]+(?:\+)?(?:\s*|[-_])*(?:people|persons?|characters?|kids?|children)",
        r"multi(?:ple)?(?:\s*|[-_])*(?:people|persons?|characters?|kids?|children)",
        r"solo", r"pair", r"couple", r"duo", r"twosome", r"threesome", r"foursome",
        r"trio", r"quartet", r"quintet", r"sextet", r"septet", r"octet", r"nonet",
        r"group", r"crowd", r"team", r"party",
    ]) +
    r")\b"
)

_URINATION_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"pee", r"peeings?", r"peeing", r"pee\s*fetish", r"pee\s*play",
        r"urination", r"urinate", r"urinating", r"urinated",
        r"piss", r"pissing", r"wetting",
    ]) +
    r")\b"
)

_SEX_ACT_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"sex", r"sexual\s+intercourse", r"having\s+sex",
        r"group(?:\s+|[-_])sex", r"gangbangs?", r"gang(?:\s+|[-_])bangs?",
    ]) +
    r")\b"
)

_CENSORING_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"censored", r"uncensored",
        r"mosaic\s+censor(?:ing|ship)?", r"mosaic\s+censored",
        r"pixelated\s+censor(?:ing|ship)?",
        r"bar\s+censor(?:ing|ship)?", r"beam\s+censor(?:ing|ship)?",
        r"light\s+censor(?:ing|ship)?",
    ]) +
    r")\b"
)

_PANTIES_PATTERN = re.compile(r"\b(panties|panty)\b")

_BRA_PATTERN = re.compile(
    r"\b(" +
    r"|".join([
        r"bra", r"brassiere", r"bralette", r"sports bra", r"strapless bra",
        r"lingerie bra", r"lace bra", r"underwire bra",
    ]) +
    r")\b"
)

def _normalize(tag: str) -> str:
    return tag.strip().replace("_", " ").lower()

_ARTIST_ALIAS_PATTERN = re.compile(r"\s*\([^()]*\)\s*$")

def _normalize_artist(artist: str) -> str:
    """Normalize artist tags and drop trailing alias hints like name_(alias)."""
    norm = _normalize(artist)
    cleaned = _ARTIST_ALIAS_PATTERN.sub("", norm).strip()
    return cleaned if cleaned else norm

def _artist_phrase(artists: List[str], p: float = 2.0) -> str:
    """
    n == 1: 返回 1 个
    n > 1 : 从 1..min(1,n) 中按权重 k**p 随机选取返回个数 k，再随机抽取 k 个
    p > 0 越大越偏向返回更多个
    """
    if random.random() < 0.05:
        return ""  # 5% 概率不加画师标签
    if not artists:
        return ""

    n = len(artists)
    if n == 1:
        names = [_normalize_artist(artists[0])]
    else:
        k_max = min(1, n)
        ks = list(range(1, k_max + 1))
        if p <= 0:
            p = 1.0  # 防御式：非正则退回线性权重
        weights = [k ** p for k in ks]
        k = random.choices(ks, weights=weights, k=1)[0]
        picked = random.sample(artists, k)
        names = [_normalize_artist(a) for a in picked]

    if len(names) == 1:
        return f"by {names[0]}"
    elif len(names) == 2:
        return f"by {names[0]} and {names[1]}"
    else:
        return "by " + ", ".join(names[:-1]) + f", and {names[-1]}"

def _rating_phrase(ratings: List[str], max_ratings: int = 1) -> str:
    if not ratings:
        return ""
    if random.random() < 0.1:
        return ""  # 10% 概率不加 rating 标签
    picked = []
    seen = set()
    for tag in ratings:
        norm = _normalize(tag)
        if not norm or norm in seen:
            continue
        if norm.startswith("rating "):
            suffix = norm.split(" ", 1)[1] if " " in norm else ""
            norm = f"rating:{suffix}" if suffix else "rating"
        else:
            norm = f"rating:{norm}"
        picked.append(norm)
        seen.add(norm)
        if len(picked) >= max_ratings:
            break
    return ", ".join(picked)

def _character_phrase(characters: List[str], max_chars: int = 5, joiner: str = ", ") -> str:
    """
    将 Camie/booru 风格的角色标签转成自然短语。
    - 默认只取前 5 个（`max_chars=5`）
    - 规则：下划线->空格，去空格；不强制加前缀（如 'character '），更贴近常见提示。
    """
    if not characters:
        return ""
    if random.random() < 0.1:
        return ""  # 10% 概率不加角色标签
    # 去重并标准化
    norm = []
    seen = set()
    for c in characters:
        t = _normalize(c)
        if not t or t in seen:
            continue
        seen.add(t)
        norm.append(t)
    if not norm:
        return ""
    # 随机打乱顺序
    norm = random.sample(norm, len(norm))
    return joiner.join(norm[:max_chars])


def _era_tag(year_tags: List[str]) -> str:
    if not year_tags:
        return ""
    if random.random() < 0.2:
        return ""  # 20% 概率不加年代标签
    years = []
    for tag in year_tags:
        match = re.search(r"(\d{4})", tag)
        if not match:
            continue
        try:
            years.append(int(match.group(1)))
        except ValueError:
            continue
    if not years:
        return ""
    latest_year = max(years)
    if latest_year < 2010:
        label = "old"
    elif latest_year < 2015:
        label = "modern"
    elif latest_year < 2020:
        label = "recent"
    else:
        label = "newest"
    return f"era:{label}"

# 互斥 / 模式限额（pattern caps）
# 说明：
# - 这些不是“分类”，而是为一组近义/同槽位标签设置“每条 caption 仅取 ≤1~2 个”的限额，避免堆砌。
# - 覆盖发色/发长/发型、眼色、胸围、衣物款式、腿部着装、鞋履、视角裁切等，在 Camie 的 general 里很常见。:contentReference[oaicite:1]{index=1}
_PATTERNS = [
    # # 发色（Camie metadata 常见 color hair 标签）：仅选 1 个
    # ("hair_color",   lambda t: _is_color_tag(t, "hair")),
    # # 发长（long/shoulder length/waist length 等）：仅 1 个
    # ("hair_length",  lambda t: bool(_HAIR_LENGTH_PATTERN.search(t))),
    # # 发型（braid/twintails/bun/pixie cut 等）：仅 1 个
    # ("hairstyle",    lambda t: bool(_HAIRSTYLE_PATTERN.search(t))),
    # # 眼睛颜色（Camie metadata 常见 color eyes 标签）：仅选 1 个
    # ("eyes_color",   lambda t: _is_color_tag(t, "eyes")),
    # # 胸围尺寸（flat/small/large/huge 等）：仅 1 个
    # ("bust_size",    lambda t: bool(_BUST_PATTERN.search(t))),
    # # 上衣袖长（sleeveless/short sleeves/off-shoulder 等）：仅 1 个
    # ("sleeves",      lambda t: bool(_SLEEVE_PATTERN.search(t))),
    # # 连衣裙/裙装（dress/skirt/gown 等）：仅 1 个
    # ("dress_skirt",  lambda t: bool(_DRESS_SKIRT_PATTERN.search(t))),
    # # 腿部着装（thighhighs/knee highs/leggings/stockings/pantyhose/bare legs 等）：仅 1 个
    # ("legwear",      lambda t: bool(_LEGWEAR_PATTERN.search(t))),
    # # 鞋履（boots/heels/sandals/sneakers/barefoot 等）：仅 1 个
    # ("footwear",     lambda t: bool(_FOOTWEAR_PATTERN.search(t))),
    # # 蝴蝶结/丝带：仅 1 个
    # ("ribbon",       lambda t: bool(_RIBBON_PATTERN.search(t))),
    # 人数（1girl/2girls 等按性别区分；solo/group 等归入通用）：各自仅 1 个
    # ("person_count_f", lambda t: bool(_PERSON_COUNT_F_PATTERN.search(t))),
    # ("person_count_m", lambda t: bool(_PERSON_COUNT_M_PATTERN.search(t))),
    # ("person_count_generic", lambda t: bool(_PERSON_COUNT_GENERIC_PATTERN.search(t))),
    # # 排尿相关（pee/peeing/urination 等）：仅 1 个
    # ("urination",    lambda t: bool(_URINATION_PATTERN.search(t))),
    # # 性行为相关（sex/gangbang/group sex 等）：仅 1 个
    # ("sex_act",     lambda t: bool(_SEX_ACT_PATTERN.search(t))),
    # 打码情况（censored/mosaic censoring 等）：仅 1 个
    # ("censoring",    lambda t: bool(_CENSORING_PATTERN.search(t))),
    # # 内裤相关（panties/side-tie panties 等）：仅 1 个
    # ("panties",      lambda t: bool(_PANTIES_PATTERN.search(t))),
    # # 胸罩相关（bra/brassiere/sports bra 等）：仅 1 个
    # ("bra",          lambda t: bool(_BRA_PATTERN.search(t))),
    # # 包/挎包：仅 1 个
    # ("bag",          lambda t: bool(_BAG_PATTERN.search(t))),
    # # 眼镜类：仅 1 个
    # ("glasses",      lambda t: bool(_GLASSES_PATTERN.search(t))),
    # # 头饰：仅 1 个
    # ("headwear",     lambda t: bool(_HEADWEAR_PATTERN.search(t))),
    # # 构图裁切：仅 1 个
    # ("body_view",    lambda t: bool(_BODY_VIEW_PATTERN.search(t))),
    # # 视角方向：仅 1 个
    # ("camera_angle", lambda t: bool(_CAMERA_ANGLE_PATTERN.search(t))),
    # # 姿态：仅 1 个
    # ("pose_basic",   lambda t: bool(_POSE_PATTERN.search(t))),
]

# 黑名单（不进训练 caption 的“元信息 / 平台工艺 / 文件属性”）
# 说明：Camie 的 meta/year 等类别用于分析很有用，但**不适合作为生成提示**；
# rating 标签单独作为锚点处理，此处不剔除。:contentReference[oaicite:2]{index=2}
_BLACKLIST = {
    # 背景/导出
    # # "simple background", "transparent background", "white background", "gradient background",
    # "sprite", "tachi-e", "official art", "game cg",
    # # 文件/画质/站点痕迹
    # "watermark", "logo", "signature", "jpeg artifacts", "upscaled", "vector",
    # "lowres", "highres", "absurdres", "incredibly absurdres", "huge filesize",
    # # 文本/翻译/字幕
    # "translated", "check translation", "speech bubble", "text focus",
    # # 其他不建议进入提示的“分类标签”
    # "general", "sensitive", "questionable", "explicit"
    # 年份序列（如 year_2017）：训练提示里通常无意义
    # 你也可以用正则整体拦截：^year_\d{4}$
}


def _trim_to_budget(text: str, budget: int) -> str:
    if _tok_len(text) <= budget:
        return text
    # 逗号/空格为边界，从后部收缩
    parts = re.split(r"\s*,\s*|\s+", text)
    kept = []
    for w in parts:
        cand = ", ".join(kept + [w]) if kept else w
        if _tok_len(cand) <= budget:
            kept.append(w)
        else:
            break
    return ", ".join(kept)

def _clean_nl(text: str) -> str:
    s = text.strip()
    # 去掉口语/摄影类废话
    s = re.sub(r"\b(an?|the)\s+(photo|image|picture)\s+of\b", "", s, flags=re.I)
    s = re.sub(r"\b(high quality|professional|photograph|camera|lens|shutter|bokeh)\b", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip(",. ").strip()
    s = s.replace(" .", ".").replace(" ,", ",")
    s = s.replace(" and ", ", ")
    return s

def generate_phrase_variants(
    general_tags: List[str],
    artists: List[str],
    k: int = 6,
    token_budget: int = 70,
    dropout: float = 0.15,
    max_general_per_variant: int = 10,
    head_keep: int = 24,
    characters: Optional[List[str]] = None,
    ratings: Optional[List[str]] = None,
    years: Optional[List[str]] = None,
    max_chars: int = 5,
) -> List[str]:
    # 预清洗 general
    g0 = []
    for tg in general_tags:
        t = _normalize(tg)
        if not t or t in _BLACKLIST:
            continue
        g0.append(t)

    head = g0[:head_keep]
    tail = g0[head_keep:]

    variants = []
    for _ in range(max(k, 1)):
        # 1) 固定锚点：角色 + 艺术家（优先占预算）
        parts = []
        char_phrase = _character_phrase(characters or [], max_chars=max_chars)
        if char_phrase:
            parts.append(char_phrase)
        rating_phrase = _rating_phrase(ratings or [], max_ratings=1)
        artist = _artist_phrase(artists)
        if artist:
            parts.append(artist)
        if rating_phrase:
            parts.append(rating_phrase)

        anchor_parts = parts[:]
        current = ", ".join(anchor_parts)
        while anchor_parts and _tok_len(current) > token_budget:
            anchor_parts.pop()  # 按顺序移除尾部锚点（优先保留角色/评级）
            current = ", ".join(anchor_parts)

        # 2) 候选：头部保序 + 尾部洗牌
        cand = head + (random.sample(tail, k=len(tail)) if tail else [])

        used_groups = set()
        general_added = 0

        # 3) 逐个尝试加入 general（随机丢弃 + 互斥 + 预算）
        for t in cand:
            if general_added >= max_general_per_variant:
                break
            if random.random() < dropout:
                continue
            # 互斥：同槽位只取一次
            if any(name in used_groups and fn(t) for name, fn in _PATTERNS):
                continue

            proposal = t if not current else f"{current}, {t}"
            if _tok_len(proposal) <= token_budget:
                current = proposal
                general_added += 1
                for name, fn in _PATTERNS:
                    if fn(t):
                        used_groups.add(name)

        # 兜底：没有任何内容时，取首个 general（若能放下）
        if not current and g0:
            t = g0[0]
            if _tok_len(t) <= token_budget:
                current = t

        era_phrase = _era_tag(years or [])
        if era_phrase:
            proposal = era_phrase if not current else f"{current}, {era_phrase}"
            if _tok_len(proposal) <= token_budget:
                current = proposal

        if current:
            variants.append(current)

    # 去重
    uniq, seen = [], set()
    for v in variants:
        if v not in seen:
            uniq.append(v); seen.add(v)
    return uniq

def generate_variants_with_nl_list(
    general_tags: List[str],
    artists: List[str],
    k: int = 6,
    token_budget: int = 70,
    phrase_ratio: float = 1.0,       # 约 80% 短语式 + 20% 自然语言
    dropout: float = 0.15,
    max_general_per_variant: int = 10,
    head_keep: int = 24,
    characters: Optional[List[str]] = None,
    ratings: Optional[List[str]] = None,
    years: Optional[List[str]] = None,
    nl_texts:  Optional[List[str]] = None,
    seed: Optional[int] = None,
    cfg_dropout: float = 0.0,
) -> List[str]:
    """
    general_tags: 已按权重降序
    artists:      画师列表，已按权重降序
    k:            每张图生成多少个变体（>=5）
    token_budget: 近似 token 上限（SDXL 建议 < 77）
    dropout:      general 的随机丢弃率（制造差异）
    max_general_per_variant: 每条最多放多少个 general 短语
    head_keep:    头部保序的候选数量（尾部会洗牌）
    cfg_dropout:  返回空 prompt 的概率（用于 CFG drop）
    """
    if seed is not None:
        random.seed(seed)

    # 1) 先生成短语式
    n_phrase = max(1, int(round(k * phrase_ratio)))
    phrase_caps = generate_phrase_variants(
        general_tags, artists, k=n_phrase, token_budget=token_budget,
        dropout=dropout, max_general_per_variant=max_general_per_variant, head_keep=head_keep,
        characters=characters, ratings=ratings, years=years
    )

    # 2) 从 nl_texts 取若干，自然语言变体
    n_nl = max(0, k - len(phrase_caps))
    nl_caps = []
    if n_nl > 0 and nl_texts:
        # 先打乱，避免总拿前几条
        pool = nl_texts[:]
        random.shuffle(pool)
        for txt in pool:
            if len(nl_caps) >= n_nl:
                break
            s = _clean_nl(txt)
            if not s:
                continue
            # 与 artist 锚定（50/50 放头或尾）
            anchors = []
            char_anchor = _character_phrase(characters or [], max_chars=5)
            if char_anchor:
                anchors.append(char_anchor)
            art = _artist_phrase(artists)
            if art:
                anchors.append(art)
            rating_anchor = _rating_phrase(ratings or [], max_ratings=1)
            if rating_anchor:
                anchors.append(rating_anchor)
            if anchors:
                anchor_str = ", ".join(anchors)
                s = f"{anchor_str}, {s}" if random.random() < 0.5 else f"{s}, {anchor_str}"

            era_phrase = _era_tag(years or [])
            if era_phrase:
                candidate = f"{s}, {era_phrase}"
                if _tok_len(candidate) <= token_budget:
                    s = candidate

            s = _trim_to_budget(s, token_budget)
            if s:
                nl_caps.append(s)

    # 3) 合并去重；若不足 k，再用短语式补足
    mixed = phrase_caps + nl_caps
    uniq, seen = [], set()
    for v in mixed:
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    while len(uniq) < k:
        extra = generate_phrase_variants(
            general_tags, artists, k=1, token_budget=token_budget,
            dropout=dropout, max_general_per_variant=max_general_per_variant, head_keep=head_keep,
            characters=characters, ratings=ratings, years=years
        )
        if not extra: break
        if extra[0] not in seen:
            uniq.append(extra[0]); seen.add(extra[0])
    results = []
    drop_rate = max(0.0, min(1.0, cfg_dropout or 0.0))
    for v in uniq[:k]:
        if drop_rate > 0.0 and random.random() < drop_rate:
            results.append("")
        else:
            results.append(v)
    return results

# ===== 用法示例 =====
if __name__ == "__main__":
    artists = ["shiratama_(shiratamaco)","kobuichi"]
    characters = []
    rating = ["rating_general"]
    general = [
        "nipples", "explicit", "breasts", "pee", "1girl", "naked apron", "peeing", "apron",
        "pink apron", "barefoot", "armpits", "open mouth", "solo", "blush",
        "spread legs", "long hair", "pussy", "large breasts", "arms behind head",
        "lying", "indoors", "on back", "mosaic censoring", "sweat", "censored", "hairband",
        "hair ribbon", "grey hair", "purple eyes", "feet", "arms up", "smile", "plant", "profile", "sidelocks", "ribbon"
    ]
    years = ["year_2019","year_2018","year_2017"]
    nl_list = [
        # "a girl with long hair in a white sundress, looking at the viewer, gentle expression",
        # "full-body illustration, ribbon details and sandals, summer vibe",
        # "standing pose with braided hair, subtle blush and purple eyes"
    ]
    caps = generate_variants_with_nl_list(
        general, artists, k=10, phrase_ratio=1, token_budget=72,
        head_keep=14, max_general_per_variant=18,
        characters=characters, ratings=rating, years=years
    )
    for i, c in enumerate(caps, 1):
        print(f"{i}. {c}")
    # caps = generate_variants_with_nl_list(
    #     general, artists, nl_list, k=1, phrase_ratio=1.00, token_budget=70, max_general_per_variant=20
    # )
    # for i, c in enumerate(caps, 1):
    #     print(f"{i}. {c}")
