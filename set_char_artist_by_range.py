#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys, tempfile, os, glob, csv
from typing import List, Tuple, Optional

# 需要修改的区间（目录ID, 起始序号, 结束序号，角色，画师，均闭区间）
RANGES: List[Tuple[int, int, int, str, str]] = [
    # For filter only
    (2883316, 711, 2000, None, "filter_invalid"),
    (885663, 2, 11, None, "filter_invalid"),
    (3457657, 1733, 1820, None, "filter_invalid"),
    (3457658, 1460, 1547, None, "filter_invalid"),
    (3417336, 1, 2000, None, "suzumori"),
    (3417337, 1, 2000, None, "suzumori"),
    (3417347, 1, 2000, None, "suzumori"),
    (3417364, 1, 2000, None, "suzumori"),
    (3417365, 1, 2000, None, "suzumori"),
    # clover day's
    (727768, 345, 461, "rindo_tsubame", "chikotam"),
    (727768, 630, 647, "rindo_tsubame", "chikotam"),
    (727768, 462, 579, "yuibashi_izumi", "chikotam"),
    (727768, 648, 663, "yuibashi_izumi", "chikotam"),
    (727768, 3, 25, "takakura_anzu", "primil"),
    (727768, 31, 106, "takakura_anzu", "primil"),
    (727768, 580, 598, "takakura_anzu", "primil"),
    (727768, 107, 220, "takakura_anri", "primil"),
    (727768, 599, 609, "takakura_anri", "primil"),
    (727768, 221, 344, "kagami_hekiru", "narumi yuu"),
    (727768, 610, 629, "kagami_hekiru", "narumi yuu"),

    (688336, 34, 147, "takakura_anri", "primil"),
    (688336, 173, 195, "takakura_anzu", "primil"),
    (688336, 201, 276, "takakura_anzu", "primil"),
    (688336, 313, 436, "kagami_hekiru", "narumi yuu"),
    (688336, 465, 583, "yuibashi_izumi", "chikotam"),
    (688336, 625, 741, "rindo_tsubame", "chikotam"),
    # Ouchi ni Kaeru made ga Marshmallow desu
    (1093883, 11, 538, "kasukabe kanon", "sasorigatame"),
    (1093883, 539, 1129, "raiha raikkonen", "ashisyun"),
    (1093883, 1130, 1829, "misuzu sasa", "chikotam"),
    (1093883, 1867, 2000, "asaka ushio", "sasorigatame"),
    (1093884, 2, 434, "asaka ushio", "sasorigatame"),
    (1113364, 1, 104, None, "ashisyun"),
    (1113364, 105, 264, "kasukabe kanon", "sasorigatame"),
    (1113364, 332, 592, "raiha raikkonen", "ashisyun"),
    (1113364, 593, 872, "misuzu sasa", "chikotam"),
    (1113364, 997, 1198, "asaka ushio", "sasorigatame"),
    (1113361, 107, 266, "kasukabe kanon", "sasorigatame"),
    (1113361, 338, 598, "raiha raikkonen", "ashisyun"),
    (1113361, 599, 878, "misuzu sasa", "chikotam"),
    (1113361, 1004, 1206, "asaka ushio", "sasorigatame"),
    # Maho x Roba -Witches Spiritual Home- 
    (1146404, 12, 13, "kujou_shizuru", "shiramori yuse"),
    (1146404, 14, 196, "konata_konatsu", "kimishima ao"),
    (1146404, 197, 406, "yamabuki_kuon", "kimishima ao"),
    (1146404, 407, 582, "kujou_shizuru", "shiramori yuse"),
    (1146404, 891, 1017, "mikage_(maho_x_roba)", "shiramori yuse"),
    (1146404, 583, 773, "hoshikawa_teru", "nanaroba hana"),
    (1146404, 774, 890, "ennis_yutoria", "nanaroba hana"),
    (1409248, 1, 2000, "konata_konatsu", None),

    (1340509, 1, 66, "konata_konatsu", "kimishima ao"),
    (1340509, 67, 174, "yamabuki_kuon", "kimishima ao"),
    (1340509, 175, 274, "kujou_shizuru", "shiramori yuse"),
    (1340509, 275, 338, "hoshikawa_teru", "nanaroba hana"),
    (1340509, 339, 386, "ennis_yutoria", "nanaroba hana"),
    (1340509, 387, 434, "mikage_(maho_x_roba)", "shiramori yuse"),
    # Koi Kakeru Shinai Kanojo
    (3000014, 35, 137, "himeno sena", "kimishima ao"),
    (3000014, 140, 219, "shindou_ayane", "kurasawa moko"),
    (3000014, 220, 324, "komari yui", "shiratama"),
    (3000014, 325, 434, "shijou_rinka", "kurasawa moko"),
    # Amairo Chocolata
    (2746430, 2, 7, "amamiya_mikuri", "korie riko"),
    (2746430, 8, 12, "yukimura_chieri", "shiratama"),
    (2746430, 13, 20, "misono_ichika", "shiratama"),
    (2746430, 21, 28, "maiba_nana", "korie riko"),
    (2746430, 29, 36, "momose_kaguya", "shiratama"),
    (2746430, 37, 74, "momose_mitsuki", "shiratama"),
    (2746430, 75, 129, "kohana_(amairo_chocolata)", "korie riko"),
    (2746430, 135, 144, "momose_mitsuki", "shiratama"),
    (2746430, 145, 154, "kohana_(amairo_chocolata)", "korie riko"),
    (2746430, 160, 171, "momose_mitsuki", "shiratama"),
    (2746430, 172, 183, "kohana_(amairo_chocolata)", "korie riko"),
    (2746430, 201, 218, "amamiya_mikuri", "korie riko"),
    (2746430, 219, 237, "yukimura_chieri", "shiratama"),
    (2746430, 238, 259, "misono_ichika", "shiratama"),
    (2746430, 260, 277, "maiba_nana", "korie riko"),
    (2746430, 278, 295, "momose_kaguya", "shiratama"),
    (2746430, 296, 340, "momose_mitsuki", "shiratama"),
    (2746430, 341, 384, "kohana_(amairo_chocolata)", "korie riko"),
    (1920176, 2, 9, "momose_mitsuki, momose_kaguya", "shiratama"),
    (1920176, 10, 15, "amamiya_mikuri", "korie riko"),
    (1920176, 16, 17, "yukimura_chieri", "shiratama"),
    (1920176, 18, 66, "misono_ichika", "shiratama"),
    (1920176, 67, 115, "maiba_nana", "korie riko"),
    (1920176, 116, 155, "momose_kaguya", "shiratama"),
    (1920176, 156, 163, "momose_mitsuki, momose_kaguya", "shiratama"),
    (1920176, 164, 212, "misono_ichika", "shiratama"),
    (1920176, 213, 258, "maiba_nana", "korie riko"),
    (1920176, 259, 298, "momose_kaguya", "shiratama"),
    (1920176, 299, 300, "amamiya_mikuri, yukimura_chieri", "shiratama, korie riko"),
    (1920176, 301, 320, "momose_mitsuki, maiba_nana", "shiratama, korie riko"),
    (1920176, 321, 380, "misono_ichika", "shiratama"),
    (1920176, 381, 428, "maiba_nana", "korie riko"),
    (1920176, 429, 474, "momose_kaguya", "shiratama"),
    (1562101, 4, 49, "amamiya_mikuri", "korie riko"),
    (1562101, 50, 89, "yukimura_chieri", "shiratama"),
    (1562101, 90, 106, "amamiya_mikuri, yukimura_chieri", "shiratama, korie riko"),
    (1562101, 107, 154, "amamiya_mikuri", "korie riko"),
    (1562101, 155, 196, "yukimura_chieri", "shiratama"),
    (2789734, 1, 473 , "maiba_nana", "korie riko"),
    (2789734, 474, 873 , "momose_mitsuki", "shiratama"),
    # Sakura no Kumo * Scarlet no Koi
    (1740860, 16, 50, "chief_(sakura_no_kumo)", "korie riko"),
    (1740860, 51, 80, "shiraide_touko", "korie riko"),
    (1740860, 81, 118, "melissa", "korie riko"),
    (1740860, 119, 158, "yukishiro_hina", "korie riko"),
    (1740860, 209, 271, "chief_(sakura_no_kumo)", "korie riko"),
    (1740860, 272, 320, "shiraide_touko", "korie riko"),
    (1740860, 321, 371, "melissa", "korie riko"),
    (1740860, 372, 418, "yukishiro_hina", "korie riko"),

    (1858102, 2, 348, "chief_(sakura_no_kumo)", "korie riko"),
    (1858102, 349, 639, "shiraide_touko", "korie riko"),
    (1858102, 1336, 1512, "melissa", "korie riko"),
    (1858102, 1567, 1767, "yukishiro_hina", "korie riko"),
    # Amazing Grace
    (1323218, 7, 48, "yune_(amazing_grace)", "korie riko"),
    (1323218, 49, 93, "kirie_(amazing_grace)", "korie riko"),
    (1323218, 94, 128, "kotoha_(amazing_grace)", "korie riko"),
    (1323218, 129, 176, "sakuya_(amazing_grace)", "korie riko"),
    (1323218, 196, 248, "yune_(amazing_grace)", "korie riko"),
    (1323218, 249, 303, "kirie_(amazing_grace)", "korie riko"),
    (1323218, 304, 358, "kotoha_(amazing_grace)", "korie riko"),
    (1323218, 359, 412, "sakuya_(amazing_grace)", "korie riko"),
    # Ikinari Anata ni Koishiteiru
    (392392, 2, 8, "yukawa_ryou", "karory"),
    (392392, 9, 18, "yanagise_eika", "korie riko"),
    (392392, 19, 25, "onigase_tane", "karory"),
    (392392, 26, 33, "yourou_tsumugu", "korie riko"),

    (392505, 8, 114, "yanagise_eika", "korie riko"),
    (392505, 115, 212, "yukawa_ryou", "karory"),
    (392505, 213, 380, "yourou_tsumugu", "korie riko"),
    (392505, 381, 514, "onigase_tane", "karory"),

    (396653, 7, 113, "yanagise_eika", "korie riko"),
    (396653, 114, 213, "yukawa_ryou", "karory"),
    (396653, 214, 399, "yourou_tsumugu", "korie riko"),
    (396653, 404, 537, "onigase_tane", "karory"),

    (1433340, 3, 103, "yukawa_ryou", "korie riko"),
    (1433340, 104, 211, "yanagise_eika", "korie riko"),
    (1433340, 212, 346, "onigase_tane", "karory"),
    (1433340, 347, 535, "yourou_tsumugu", "korie riko"),
    # Kotoba no Kieta Hi 
    (799921, 2, 105, "mizukashi_aoi", "motoyon"),
    (799921, 106, 261, "arisugawa_azusa", "motoyon"),
    (799921, 287, 307, "mizukashi_aoi, arisugawa_azusa", "motoyon"),
    # Hoshikoi Twinkle
    (1023049, 16, 232, "narusegawa_nagi", "korie riko"),
    (1023049, 233, 374, "kagami_sakura", "korie riko"),
    (1023049, 397, 588, "murakumo_soraha", "korie riko"),
    (1023049, 589, 754, "origami_tamaki", "korie riko"),

    (1859635, 2, 680, "murakumo_soraha", "korie riko"),
    (1859635, 752, 1347, "narusegawa_nagi", "korie riko"),
    (1859635, 1378, 2000, "kagami_sakura", "korie riko"),
    (1859671, 1, 16, "kagami_sakura", "korie riko"),
    (1859671, 52, 550, "origami_tamaki", "korie riko"),
    # Neko to Wakai Seyo!
    (2231256, 2, 155, "yasaka_kanon", "koruri"),
    (2231256, 156, 319, "tachibana_seika", "koruri"),
    (2231256, 320, 487, "momiji_akira", "koruri"),
    (2231256, 488, 650, "hiba_mitsuki", "koruri"),
    (2231256, 651, 734, "yasaka_hiiragi", "koruri"),

    (3425833, 1, 250, "yasaka_kanon", "koruri"),
    (3425833, 251, 536, "tachibana_seika", "koruri"),
    (3425833, 537, 770, "momiji_akira", "koruri"),
    (3425833, 771, 1145, "hiba_mitsuki", "koruri"),
    (3425833, 1146, 11512850, "yasaka_hiiragi", "koruri"),

    # Koi Kakeru Shinai Kanojo
    (868607, 39, 143, "himeno sena", "kimishima ao"),
    (868607, 144, 223, "shindou_ayane", "kurasawa moko"),
    (868607, 224, 328, "komari yui", "shiratama"),
    (868607, 329, 438, "shijou_rinka", "kurasawa moko"),
    (868607, 439, 640, "himeno sena", "kimishima ao"),
    (868607, 641, 855, "shindou_ayane", "kurasawa moko"),
    (868607, 856, 1090, "komari yui", "shiratama"),
    (868607, 1091, 1326, "shijou_rinka", "kurasawa moko"),
    # miagete goran yoru no hoshi o
    (1245707, 13, 23, "amanogawa_saya", "motoi ayumu"),
    (1245707, 48, 72, "amanogawa_saya", "motoi ayumu"),
    (1245707, 107, 130, "amanogawa_saya", "motoi ayumu"),
    (1245707, 305, 318, "amanogawa_saya", "motoi ayumu"),
    (943537, 3, 184, "houkiboshi_hikari", "yashima takahiro"),
    (943537, 185, 294, "kusakabe_korona", "yashima takahiro"),
    (943537, 295, 403, "saotome_miharu", "yashima takahiro"),
    (943537, 404, 617, "shiratori_orihime", "motoi ayumu"),
    (943537, 618, 837, "amanogawa_saya", "motoi ayumu"),
    (900491, 304, 657, "houkiboshi_hikari", "yashima takahiro"),
    (900491, 658, 957, "kusakabe_korona", "yashima takahiro"),
    (900491, 958, 1282, "shiratori_orihime", "motoi ayumu"),
    (900491, 1283, 1540, "amanogawa_saya", "motoi ayumu"),
    # Ichiban Janakya Dame Desu ka?
    (634594, 11, 13, "futaba_hisui", "nanase meruchi"),
    (634594, 17, 19, "futaba_hisui", "nanase meruchi"),
    (634594, 20, 40, "ichimine_kohaku", "nanase meruchi"),
    (634594, 45, 106, "ichimine_kohaku", "nanase meruchi"),
    (634594, 107, 220, None, "kinta"),
    (634594, 221, 292, "futaba_hisui", "nanase meruchi"),
    (634594, 367, 369, "futaba_hisui", "nanase meruchi"),
    (847794, 1, 418, "natsuki rino", None),
    (979189, 1, 53, "mito kohaku", None),
    (979189, 54, 111, "saijou hifumi", None),
    (979189, 228, 250, "mito kohaku", None),
    (979189, 251, 271, "saijou hifumi", None),

    (1056434, 22, 32, "mito kohaku", None),
    (1056434, 33, 43, "saijou hifumi", None),
    (1056434, 54, 65, "yunohana nano", None),
    (1056434, 91, 132, "mito kohaku", None),
    (1056434, 133, 172, "saijou hifumi", None),
    (1056434, 211, 247, "yunohana nano", None),

    (1993857, 49, 71, "mito kohaku", None),
    (1993857, 72, 92, "saijou hifumi", None),
    (1993857, 119, 137, "yunohana nano", None),
    (1993857, 138, 141, "yunohana nano, mito kohaku", None),
    (1993857, 143, 193, "mito kohaku", None),
    (1993857, 194, 251, "saijou hifumi", None),
    (1993857, 312, 367, "yunohana nano", None),

    # Magical Marriage Lunatics!!
    (633524, 2, 141, "luce yami asutarite", "yamakaze ran"),
    (633524, 142, 277, "julia lin road", "sakurazaka tsuchiyu"),
    (633524, 278, 405, "mitsu no tama yori hime", "yamakaze ran"),
    (633524, 406, 553, "amagi karin", "yamakaze ran"),
    (633524, 554, 671, "luluna", "yamakaze ran"),
    (633524, 672, 815, "shirahase yuuna", "yamakaze ran"),

    (1217027, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (1217027, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (1217027, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (1217027, 407, 554, "amagi karin", "yamakaze ran"),
    (1217027, 555, 672, "luluna", "yamakaze ran"),
    (1217027, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (634833, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (634833, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (634833, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (634833, 407, 554, "amagi karin", "yamakaze ran"),
    (634833, 555, 672, "luluna", "yamakaze ran"),
    (634833, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (3466476, 1, 672, "luce yami asutarite", "yamakaze ran"),
    (3466476, 673, 1909, "julia lin road", "sakurazaka tsuchiyu"),
    (3466477, 1, 600, "amagi karin", "yamakaze ran"),
    (3466477, 605, 1207, "mitsu no tama yori hime", "yamakaze ran"),
    (3466478, 1, 832, "luluna", "yamakaze ran"),
    (3466478, 833, 1344, "shirahase yuuna", "yamakaze ran"),

    # Himawari!! -Anata Dake wo Mitsumeteru-
    (735531, 27, 153, "onii_mikoto", "p19"),
    (735531, 154, 273, "teshigawara_tsubasa", "p19"),
    (735531, 274, 380, "mikazuki_tenma", "p19"),
    (735531, 381, 526, "wasurenagusa_izumi", "p19"),
    (735531, 527, 635, "todoroki_kei", "p19"),
    # Docchi no i ga Suki Desu ka?
    (1525889, 6, 604, None, "monoto"),
    (1525889, 605, 1190, "tanemura_koyuzu", "netarou"),
    (1525889, 1191, 1748, None, "ayase hazuki"),
    (1525889, 1749, 2000, "kaminoyama_mea", "netarou"),
    (1525937, 1, 268, "kaminoyama_mea", "netarou"),
    # Kujiragami no Tearstilla
    (885411, 6, 98, "tenkawa_mitsuki", None),
    (885411, 99, 193, "narumi_marine", None),
    (885411, 194, 290, "kamitouno_ena", None),
    (885411, 291, 411, "riru_whale", None),
    (3450027, 1, 552, "tenkawa_mitsuki", "mikagami mamizu"),
    (3450027, 553, 1262, "narumi_marine", "mikagami mamizu"),
    (3450027, 1263, 1790, "kamitouno_ena", "mikagami mamizu"),
    (3450028, 1, 670, "riru_whale", "mikagami mamizu"),
    # Hatsujou Sprinkle
    (1093463, 2, 159, "meidou_hazuki", "mikagami mamizu"),
    (1093463, 160, 297, "hanabusa_koharu", "mikagami mamizu"),
    (1093463, 298, 435, "hinata_mio", "mikagami mamizu"),
    (1093463, 436, 577, "momosaki_shizuku", "mikagami mamizu"),

    (1147899, 1, 290, "meidou_hazuki", "mikagami mamizu"),
    (1147899, 291, 566, "hanabusa_koharu", "mikagami mamizu"),
    (1147899, 567, 791, "hinata_mio", "mikagami mamizu"),
    (1147899, 792, 975, "momosaki_shizuku", "mikagami mamizu"),
    # Lunaris Filia
    (1166701, 2, 82, "sylvia_luna_infinitus", "mikagami mamizu"),
    (1166701, 83, 155, "minase_yukari", "mikagami mamizu"),
    (1166701, 156, 241, "hayama_mai", "mikagami mamizu"),
    (1166701, 242, 319, "melis", "mikagami mamizu"),
    (1166701, 320, 388, "kotoharu_kanon", "mikagami mamizu"),
    (1166701, 389, 446, "filia_(lunaris_filia)", "mikagami mamizu"),

    (1167070, 1, 750, "sylvia_luna_infinitus", "mikagami mamizu"),
    (1167070, 751, 1463, "minase_yukari", "mikagami mamizu"),
    (1167070, 1464, 1871, "hayama_mai", "mikagami mamizu"),
    (1167078, 1, 381, "melis", "mikagami mamizu"),
    (1167078, 382, 792, "kotoharu_kanon", "mikagami mamizu"),
    (1167078, 793, 1259, "filia_(lunaris_filia)", "mikagami mamizu"),

    # Neko☆Koi! 
    (879411, 4, 91, "ryuudou_misaki", "mikagami mamizu"),
    (879411, 92, 180, "kiryuu_hina", "mikagami mamizu"),
    (879411, 181, 272, "toono_sayaka", "mikagami mamizu"),
    (879411, 273, 359, "koshimizu_rei", "mikagami mamizu"),

    # Justy×Nasty
    (1088986, 2, 111, "onose_mana", "mikagami mamizu"),
    (1088986, 112, 217, "kuroki_kirie", "mikagami mamizu"),
    (1088986, 218, 320, "kamui_mikaru", "mikagami mamizu"),
    (1088986, 321, 411, "kagami_hibiki", "mikagami mamizu"),
    (1088986, 412, 496, "ootsuki_karin", "mikagami mamizu"),

    (1147801, 1, 205, "onose_mana", "mikagami mamizu"),
    (1147801, 206, 415, "kuroki_kirie", "mikagami mamizu"),
    (1147801, 416, 603, "kamui_mikaru", "mikagami mamizu"),
    (1147801, 604, 743, "kagami_hibiki", "mikagami mamizu"),
    (1147801, 744, 932, "ootsuki_karin", "mikagami mamizu"),

    # World Election
    (908667, 5, 118, "kururu_(world_election)", "mikagami mamizu"),
    (908667, 119, 216, "sophia_usty", "tenmaso"),
    (908667, 217, 324, "shishidou_iori", "ichikura robo"),
    (908667, 325, 425, "parfil", "mikagami mamizu"),
    (908667, 426, 529, "miyamoto_haruka", "r ken"),
    (908667, 530, 549, "minayo_(world_election)", None),

    # Floral Flowlove
    (960177, 3, 207, "adelheid_von_bergstrasse", "matsumiya kiseri"),
    (960177, 208, 408, "mihato kano", "hontani kanae"),
    (960177, 409, 647, "tsubaki kohane", "arisue tsukasa"),
    (960177, 648, 890, "tokisaka nanao", "toranosuke"),
    # Kiniro Loveriche -Golden Time-
    (1953903, 2, 323, "sylvia_le_cruzcrown_sortilege_sisua", "hontani kanae"),
    (1953903, 324, 542, "kisaki_reina", "toranosuke"),
    (1953903, 543, 818, "heroina_di_caballero_ista", "toranosuke"),
    (1953903, 820, 862, "souma_ria", "hontani kanae"),
    (1953903, 907, 958, "souma_ria", "hontani kanae"),
    (1953903, 1005, 1120, "kuryuu_akane", "arisue tsukasa"),
    (1953903, 1121, 1161, "caminal_le_pultaia_sortilege_sisua", "purin purin"),

    (1936004, 2, 117, "kuryuu_akane", "arisue tsukasa"),
    (1936004, 118, 347, "heroina_di_caballero_ista", "toranosuke"),
    (1936004, 348, 388, "caminal_le_pultaia_sortilege_sisua", "purin purin"),
    (1936004, 391, 605, "kisaki_reina", "toranosuke"),
    (1936004, 607, 673, "souma_ria", "hontani kanae"),
    (1936004, 675, 724, "souma_ria", "hontani kanae"),
    (1936004, 771, 1089, "sylvia_le_cruzcrown_sortilege_sisua", "hontani kanae"),

    (1157754, 2, 117, "kuryuu_akane", "arisue tsukasa"),
    (1157754, 118, 393, "heroina_di_caballero_ista", "toranosuke"),
    (1157754, 394, 434, "caminal_le_pultaia_sortilege_sisua", "purin purin"),
    (1157754, 435, 653, "kisaki_reina", "toranosuke"),
    (1157754, 655, 740, "souma_ria", "hontani kanae"),
    (1157754, 742, 792, "souma_ria", "hontani kanae"),
    (1157754, 844, 1161, "sylvia_le_cruzcrown_sortilege_sisua", "hontani kanae"),

    (1369330, 6, 36, "kuryuu_akane", "arisue tsukasa"),
    (1369330, 37, 249, "jougasaki_ayaka", "arisue tsukasa"),
    (1369330, 250, 298, "heroina_di_caballero_ista", "toranosuke"),
    (1369330, 299, 453, "caminal_le_pultaia_sortilege_sisua", "purin purin"),
    (1369330, 454, 476, "kisaki_reina", "toranosuke"),

    (2274092, 3, 33, "kuryuu_akane", "arisue tsukasa"),
    (2274092, 34, 246, "jougasaki_ayaka", "arisue tsukasa"),
    (2274092, 247, 295, "heroina_di_caballero_ista", "toranosuke"),
    (2274092, 296, 450, "caminal_le_pultaia_sortilege_sisua", "purin purin"),
    (2274092, 451, 473, "kisaki_reina", "toranosuke"),

    (1497885, 1, 2000, "jougasaki_ayaka", None),
    # KisaragiGOLD★STAR~Moonlight serenade in autum
    (713608, 58, 61, "fujimaru_mikoto", "toranosuke"),
    (713608, 3, 109, "nitta_ichika", "chimaro"),
    (713608, 117, 126, "nitta_ichika", "chimaro"),
    (713608, 127, 241, "fujimaru_mikoto", "toranosuke"),
    (713608, 242, 365, "endou_saya_(kisaragi_gold_star)", "hontani kanae"),
    (713608, 445, 573, "haotone_tsubasa", "hontani kanae"),
    # Hanayome to Maou
    (746122, 9, 67, "celica tepes lunatica", None),
    # Anata ni Koisuru Renai Recette
    (1243251, 1, 1280, "tachibana nonoka", "komeshiro kasu"),
    (1243251, 1281, 2000, "oozono yuzuki", "fummy"),
    (1243267, 1, 560, "oozono yuzuki", "fummy"),
    (1243267, 561, 1840, "kagiyoshi fuuka", "komeshiro kasu"),
    (1243267, 1841, 2000, "shirosaki mieru", "fummy"),
    (1243602, 1, 1120, "shirosaki mieru", "fummy"),
    (1067390, 23, 182, "tachibana nonoka", "komeshiro kasu"),
    (1067390, 183, 338, "oozono yuzuki", "fummy"),
    (1067390, 339, 498, "kagiyoshi fuuka", "komeshiro kasu"),
    (1067390, 499, 706, "shirosaki mieru", "fummy"),
    # Tsumi no Hikari Rendezvous Goukaban
    (913381, 1, 248, "tsubaki fuuka", "satasama"),
    (913368, 773, 1169, "tsubaki fuuka", "satasama"),
    (913368, 1844, 2000, "tsubaki fuuka", "satasama"),
    (2657576, 774, 1170, "tsubaki fuuka", "satasama"),
    (2657600, 409, 813, "tsubaki fuuka", "satasama"),
    (2657600, 1, 408, "masumi_ai", "yuzuna hiyo"),
    (2657600, 814, 1142, "misono_tsubura", "mizuno sao"),
    # Amatsutsumi
    (2285016, 4, 75, "oribe_kokoro", "koku"),
    (2285016, 119, 318, "minazuki_hotaru_(amatsutsumi)", "koku"),
    (2285016, 334, 390, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (2285016, 409, 504, "koizuka_mana_(amatsutsumi)", "koku"),
    (2285016, 570, 752, "oribe_kokoro", "koku"),
    (2285016, 753, 971, "minazuki_hotaru_(amatsutsumi)", "koku"),
    (2285016, 972, 1130, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (2285016, 1131, 1330, "koizuka_mana_(amatsutsumi)", "koku"),
    (1121856, 1, 564, "oribe_kokoro", "koku"),
    (1121856, 565, 1574, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (1121856, 1575, 2000, "koizuka_mana_(amatsutsumi)", "koku"),
    (1121870, 1, 110, "koizuka_mana_(amatsutsumi)", "koku"),
    (1121870, 111, 620, "minazuki_hotaru_(amatsutsumi)", "koku"),
    (1121870, 621, 810, None, "koku"),
    (959791, 726, 940, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (959791, 4, 258, "oribe_kokoro", "koku"),
    (959791, 959, 1254, "koizuka_mana_(amatsutsumi)", "koku"),
    (959791, 302, 725, "minazuki_hotaru_(amatsutsumi)", "koku"),
    # Chrono Clock
    (1140183, 12, 120, "dorothy_davenport", "tsukimori hiro"),
    (1140183, 121, 290, "kuro_(chrono_clock)", "koku"),
    (1140183, 291, 452, "jounouchi_makoto", "koku"),
    (1140183, 453, 558, "sawatari_michiru", "tsukimori hiro"),

    (904316, 12, 77, "dorothy_davenport", "tsukimori hiro"),
    (904316, 63, 187, "kuro_(chrono_clock)", "koku"),
    (904316, 192, 296, "jounouchi_makoto", "koku"),
    (904316, 297, 350, "sawatari_michiru", "tsukimori hiro"),
    (904316, 497, 540, "dorothy_davenport", "tsukimori hiro"),
    (904316, 541, 601, "kuro_(chrono_clock)", "koku"),
    (904316, 602, 672, "jounouchi_makoto", "koku"),
    (904316, 673, 712, "sawatari_michiru", "tsukimori hiro"),

    (1464212, 1, 155, "dorothy_davenport", "tsukimori hiro"),
    (1464212, 454, 633, "kuro_(chrono_clock)", "koku"),
    (1464151, 90, 479, "jounouchi_makoto", "koku"),
    (1464151, 480, 1109, "sawatari_michiru", "tsukimori hiro"),
    (1464151, 1366, 2000, "dorothy_davenport", "tsukimori hiro"),

    # Hapymaher
    (2564037, 4, 143, "toriumi_arisu", "koku"),
    (2564037, 312, 348, "naitou_maia", "koku"),
    (2564037, 349, 460, "hasuno_saki", "tsukimori hiro"),
    (2564037, 461, 570, "yayoi_b_lutwidge", "koku"),
    (2564037, 612, 708, "toriumi_arisu", "koku"),
    (2564037, 711, 800, "hirasaka_keiko", "tsukimori hiro"),
    (2564037, 801, 888, "hasuno_saki", "tsukimori hiro"),
    (2564037, 889, 920, "yayoi_b_lutwidge", "koku"),
    (2564037, 945, 979, "yayoi_b_lutwidge", "koku"),
    (2564037, 1125, 1201, "naitou_maia", "koku"),
    (1256674, 4, 143, "toriumi_arisu", "koku"),
    (1256674, 440, 567, "hasuno_saki", "tsukimori hiro"),
    (1256674, 188, 224, "naitou_maia", "koku"),
    (1256674, 225, 334, "yayoi_b_lutwidge", "koku"),
    (1179790, 2, 417, "toriumi_arisu", "koku"),
    (1179790, 418, 942, "hasuno_saki", "tsukimori hiro"),
    (1179790, 943, 1548, "yayoi_b_lutwidge", "koku"),

    (1179801, 27, 536, "hirasaka_keiko", "tsukimori hiro"),
    (1179801, 537, 693, "naitou_maia", "koku"),

    # Mirai Nostalgia
    (992205, 5, 37, "anna_(mirai_nostalgia)", "koku"),
    (992205, 62, 123, "anna_(mirai_nostalgia)", "koku"),
    (992205, 124, 234, None, "siki"),
    (992205, 235, 351, "kasuga_iori", "koku"),
    (992205, 358, 381, None, "siki"),
    (992205, 383, 473, None, "siki"),
    (992205, 479, 557, "hatori_uta_(mirai_nostalgia)", "koku"),

    (992205, 778, 873, "anna_(mirai_nostalgia)", "koku"),
    (992205, 874, 897, "hatori_uta_(mirai_nostalgia)", "koku"),
    (992205, 898, 923, "kasuga_iori", "koku"),
    (992205, 924, 1013, None, "siki"),

    # Soreyori no Prologue
    (793448, 1, 402, "tsuzuki_haruka", "mizuno sao"),
    (793448, 403, 2000, "himeno_towa", "yuzuna hiyo"),
    (793449, 1, 29, "himeno_towa", "yuzuna hiyo"),
    (793449, 30, 1012, "sakurai_mayura", "satasama"),
    (793449, 1013, 2000, "tsuzuki_haruka", "mizuno sao"),
    (793088, 13, 162, "himeno_towa", "yuzuna hiyo"),
    (793088, 396, 490, None, "satasama"),
    (793088, 491, 851, "himeno_towa", "yuzuna hiyo"),
    (793088, 1158, 1188, "sakurai_mayura", "satasama"),
    (793447, 1, 179, "himeno_towa", "yuzuna hiyo"),
    (793447, 1161, 1232, "himeno_towa", "yuzuna hiyo"),
    (793447, 196, 1419, "sakurai_mayura", "satasama"),
    (793447, 1421, 2000, "tsuzuki_haruka", "mizuno sao"),
    (793453, 1, 385, "tsuzuki_haruka", "mizuno sao"),
    (1536946, 1, 2000, "himeno_towa", "yuzuna hiyo"),
    (1537715, 1, 289, "tsuzuki_haruka", "mizuno sao"),
    (1537491, 1, 306, "tsuzuki_haruka", "mizuno sao"),
    (1537491, 307, 1933, "himeno_towa", "yuzuna hiyo"),
    (1537491, 1934, 2000, "sakurai_mayura", "satasama"),
    (1537567, 1, 916, "sakurai_mayura", "satasama"),
    (1537567, 917, 2000, "tsuzuki_haruka", "mizuno sao"),
    (1537478, 1, 104, "himeno_towa", "yuzuna hiyo"),
    (1537478, 1082, 1153, "himeno_towa", "yuzuna hiyo"),
    (1537478, 121, 1341, "sakurai_mayura", "satasama"),
    (1537478, 1342, 2000, "tsuzuki_haruka", "mizuno sao"),
    (1537457, 6, 380, "himeno_towa", "yuzuna hiyo"),
    (1537457, 484, 800, "himeno_towa", "yuzuna hiyo"),
    (1537457, 809, 822, "tsuzuki_haruka", "mizuno sao"),
    # Oshioki Namaiki Gal
    (3417922, 1, 2000, "natsuki rino", "karatabe"),
    (847794, 1, 2000, "natsuki rino", None),
    # Toki o Tsumugu Yakusoku
    (917393, 4, 94, "mizuno koharu", "shiwasu horio"),
    (917393, 95, 166, "usui honoka", "shiwasu horio"),
    (917393, 167, 245, "jinguu misaki", "odawara hakone"),
    (917393, 255, 347, "sawamura yui", "shiwasu horio"),
    # Koisuru Ojou-sama wa Ecchi na Hanayome
    (970538, 1, 103, "tsukimiya_asuka", "nemaki"),
    (970538, 104, 107, "maya_tsubura", "yoshi hyuuma"),
    (970538, 108, 112, "serizawa_yukino", "sousouman"),
    (970538, 113, 116, "misaka_sumire", "agekichi"),
    (970538, 117, 120, "serizawa_yukino", "sousouman"),
    (970538, 121, 124, "maya_tsubura", "yoshi hyuuma"),
    (970538, 125, 128, "miake_hiyoko", "tsuneyoshi"),
    (970538, 132, 145, "tsukimiya_asuka", "nemaki"),
    (970538, 146, 216, "miake_hiyoko", "tsuneyoshi"),
    (970538, 217, 312, "misaka_sumire", "agekichi"),
    (970538, 313, 397, "maya_tsubura", "yoshi hyuuma"),
    (970538, 398, 469, "serizawa_yukino", "sousouman"),
    (970538, 470, 488, "tsukimiya_asuka, misaka_sumire, maya_tsubura", "nemaki, agekichi, yoshi hyuuma"),
    (970538, 489, 500, "serizawa_yukino, miake_hiyoko", "sousouman, tsuneyoshi"),
    # Ecchi de Hentai! Yakimochi Ojou-sama
    (836073, 5, 182, "kurusugawa_alice", "goma satoshi"),
    (836073, 183, 410, "kagami_hibiki_(ecchi_de_hentai!)", "satofuji masato"),
    (836073, 411, 562, "shinomori_kazuha", "yoshi hyuuma"),
    (836073, 563, 717, "shiratori_mayu", "satofuji masato"),
    (836073, 718, 887, "orimiya_meru", "satofuji masato"),
    # Ubu na Otome no Ecchi na Onegai
    (1133866, 3, 116, "nanasato_nene", None),
    (1133866, 117, 229, "momozono_hozumi", None),
    (1133866, 230, 352, "hikami_ayame", None),
    (1133866, 353, 507, "anesaki_mimari", None),
    (1133866, 508, 654, "hoshimi_kano", None),

    (1135068, 1, 70, "nanasato_nene", None),
    (1135068, 71, 119, "momozono_hozumi", None),
    (1135068, 120, 154, "hikami_ayame", None),
    (1135068, 155, 189, "anesaki_mimari", None),
    (1135068, 190, 217, "hoshimi_kano", None),

    (1512874, 4, 291, "nanasato_nene", None),
    (1512874, 292, 675, "momozono_hozumi", None),
    (1512874, 676, 1011, "hikami_ayame", None),
    (1512874, 1012, 1395, "anesaki_mimari", None),
    (1512874, 1396, 1724, "hoshimi_kano", None),
    # Otome Kishi Ima Sugu Watashi o Dakishimete
    (1403922, 119, 312, "akatsuki_amane", "yashima takahiro"),
    (1403922, 313, 501, "kuromine_misumi", "haruruyuco"),
    (1403922, 502, 735, "yukishiro_chitome", "lucie"),
    # Noble Butler
    (3328263, 96, 172, "luna_fleur_renisphia", "natsuki marina"),
    (3328263, 173, 257, "melty_fleur_renisphia", "mogupon"),
    (3328263, 258, 338, "tina_fleur_renisphia", "shiwasu horio"),

    (3328263, 401, 455, "luna_fleur_renisphia", "natsuki marina"),
    (3328263, 456, 513, "melty_fleur_renisphia", "mogupon"),
    (3328263, 514, 584, "tina_fleur_renisphia", "shiwasu horio"),

    # Pure Song Garden!
    (1081513, 2, 236, "shimokuni asuka", "bekotarou"),
    (1081513, 237, 578, "hoshino iroha", "motoi ayumu"),
    (1081513, 579, 774, "kawai_kuon", "marui"),
    (1081513, 775, 942, None, " "),
    (1081513, 943, 1129, "suzu_(pure_song_garden!)", "bekotarou"),
    # Tamayura Mirai
    (1538757, 1, 168, "suishouseki midari", "matsumiya kiseri"),
    (1538757, 169, 336, "nekotenguu hanako", "ameto yuki"),
    (1538757, 337, 476, "kohaku shiro", "ameto yuki"),
    (1538757, 477, 616, "kamikaze yukina", "matsumiya kiseri"),
    (1423467, 217, 378, "suishouseki midari", "matsumiya kiseri"),
    (1423467, 13, 216, "nekotenguu hanako", "ameto yuki"),
    (1423467, 381, 528, "kohaku shiro", "ameto yuki"),
    (1423467, 551, 696, "kamikaze yukina", "matsumiya kiseri"),
    # Hanasaki Work Spring!
    (800080, 2, 243, "kuon ayano", "toranosuke"),
    (800080, 244, 450, "kotobuki hikari", "matsumiya kiseri"),
    (800080, 451, 627, "shiranui inori", "hontani kanae"),
    (800080, 628, 772, "koutsuki kanna", "arisue tsukasa"),
    (800080, 773, 920, "hanasaki nonoka", "hontani kanae"),
    (800080, 921, 1131, "soramori wakaba", "toranosuke"),
    # AMBITIOUS MISSION
    (2230428, 2, 119, "arise_kaguya", "hontani kanae"),
    (2230428, 120, 150, "arise_kaguya, arise_atena", "hontani kanae"),
    (2230428, 151, 171, "arise_kaguya", "hontani kanae"),
    (2230428, 536, 632, "arise_atena", "hontani kanae"),
    (2230428, 362, 532, "hongou_nijimu", "arisue tsukasa"),
    (2230428, 172, 361, "ishikawa_yae", "toranosuke"),
    (2562932, 4, 65, "arise_atena", "hontani kanae"),
    (2562932, 73, 139, "arise_kaguya", "hontani kanae"),
    (2595303, 160, 215, "ishikawa_yae", "toranosuke"),
    # Kakenuke Seishun Sparking!
    (1719088, 6, 198, "kohinata_hibiki", "hontani kanae"),
    (1719088, 199, 339, "hijiri_kikka", "hontani kanae"),
    (1719088, 340, 531, "kaidou_nagiko", "arisue tsukasa"),
    (1719088, 532, 645, "toono_ritsu", "hanesaki seika"),
    (1719088, 646, 791, "kashima_riri", "toranosuke"),
    (1719088, 792, 956, "hiiragi_shiori", "toranosuke"),


    (3097843, 5, 197, "kohinata_hibiki", "hontani kanae"),
    (3097843, 198, 343, "kashima_riri", "toranosuke"),
    (3097843, 344, 507, "hiiragi_shiori", "toranosuke"),
    (3097843, 508, 699, "kaidou_nagiko", "arisue tsukasa"),
    (3097843, 700, 841, "hijiri_kikka", "hontani kanae"),
    (3097843, 842, 954, "toono_ritsu", "hanesaki seika"),
    # Hatsuyuki Sakura
    (3156090, 3, 151, "azuma_yoru", "chimaro"),
    (3156090, 152, 224, "kozakai_aya", "toranosuke"),
    (3156090, 225, 273, "shirokuma_(hatsuyuki_sakura)", "hontani kanae"),
    (3156090, 302, 384, "shinonome_nozomu", "hontani kanae"),
    (3156090, 386, 493, "tamaki_sakura", "hontani kanae"),

    (1010116, 4, 109, "azuma_yoru", "chimaro"),
    (1010116, 110, 185, "kozakai_aya", "toranosuke"),
    (1010116, 186, 239, "shirokuma_(hatsuyuki_sakura)", "hontani kanae"),
    (1010116, 269, 339, "shinonome_nozomu", "hontani kanae"),
    (1010116, 341, 449, "tamaki_sakura", "hontani kanae"),

    # Glass Hime to Kagami no Juusha
    (1838084, 2, 290, "bernadette_henrietta_eiselstone", "arisue tsukasa"),
    (1838084, 291, 525, "henri-francis_la_bernard", "syroh"),
    (1838084, 526, 728, "sasara_orihime", "arisue tsukasa"),
    (1838084, 729, 916, "shirogama_naomi", "syroh"),
    (1988313, 2, 655, "henri-francis_la_bernard", "syroh"),
    (1988313, 656, 1150, "bernadette_henrietta_eiselstone", "arisue tsukasa"),
    (1988313, 1151, 1438, None, "arisue tsukasa"),
    (1988313, 1439, 1690, None, "syroh"),
    (1988313, 1691, 1798, None, "arisue tsukasa"),
    (1988313, 1799, 2000, "shirogama_naomi", "syroh"),
    (1988314, 1, 338, "shirogama_naomi", "syroh"),
    (1988314, 339, 2000, "sasara_orihime", "arisue tsukasa"),
    (1988315, 1, 267, "sasara_orihime", "arisue tsukasa"),

    (1988342, 3, 54, "sasara_orihime", "arisue tsukasa"),
    (1988342, 55, 74, "shirogama_naomi", "syroh"),
    # Karumaruka ＊ Circle
    (633332, 2, 152, "natsume_koyomi", "toranosuke"),
    (633332, 153, 315, "otone_nicole", "hontani kanae"),
    (633332, 316, 481, "amagase_natsuki", "hontani kanae"),
    (633332, 482, 576, "asahina_shin", "toranosuke"),
    (633332, 577, 620, "yukiha_kousaka", "chimaro"),
    # Primal x Hearts 2
    (1939336, 1, 126, "alicetia wallenberg kezouji", "sasorigatame"),
    (868985, 1, 127, "tsukiyono usagi", "sasorigatame"),
    (868985, 334, 589, "alicetia wallenberg kezouji", "sasorigatame"),
    (868985, 590, 620, "tenjindaira haruhi", "sasorigatame"),
    (868985, 864, 998, "kuryuu mashiro", "ashisyun"),
    (868985, 1278, 1509, "kuragano sara", "sasorigatame"),
    (868985, 1509, 1683, "tatebayashi tateha", "ashisyun"),
    (868985, 1684, 1951, "tsukiyono usagi", "sasorigatame"),
    (868985, 1952, 1978, "komagata yuzuki", "ashisyun"),
    (868991, 185, 440, "alicetia wallenberg kezouji", "sasorigatame"),
    (868991, 441, 471, "tenjindaira haruhi", "sasorigatame"),
    (868991, 715, 849, "kuryuu mashiro", "ashisyun"),
    (868991, 1131, 1304, "tatebayashi tateha", "ashisyun"),
    (868991, 1307, 1574, "tsukiyono usagi", "sasorigatame"),
    (868991, 1575, 1601, "komagata yuzuki", "ashisyun"),
    (868977, 158, 623, "alicetia wallenberg kezouji", "sasorigatame"),
    (868977, 632, 1169, "kuryuu mashiro", "ashisyun"),
    (868977, 1195, 1210, "kuragano sara", "sasorigatame"),
    (868977, 1211, 1689, "tatebayashi tateha", "ashisyun"),
    (868977, 1690, 2000, "tsukiyono usagi", "sasorigatame"),
    # Primal x Hearts
    (1183093, 39, 442, "tenjindaira haruhi", "sasorigatame"),
    (1183093, 443, 858, "kanna kana", "ashisyun"),
    (1183093, 874, 1416, "kuragano sara", "sasorigatame"),
    (1183093, 1417, 1795, "komagata yuzuki", "ashisyun"),

    (1317096, 36, 439, "tenjindaira haruhi", "sasorigatame"),
    (1317096, 440, 855, "kanna kana", "ashisyun"),
    (1317096, 871, 1413, "kuragano sara", "sasorigatame"),
    (1317096, 1414, 1792, "komagata yuzuki", "ashisyun"),

    (1183141, 1, 186, "tenjindaira haruhi", "sasorigatame"),
    (1183141, 187, 342, "kanna kana", "ashisyun"),
    (1183141, 567, 740, "kuragano sara", "sasorigatame"),
    (1183141, 798, 986, "komagata yuzuki", "ashisyun"),
    (1183141, 987, 1017, "tenjindaira haruhi", "sasorigatame"),
    (1183141, 1018, 1043, "kanna kana", "ashisyun"),
    (1183141, 1070, 1098, "kuragano sara", "sasorigatame"),
    (1183141, 1099, 1125, "komagata yuzuki", "ashisyun"),

    (868964, 612, 2000, "kuragano sara", "sasorigatame"),
    (868964, 16, 161, "kuragano sara", "sasorigatame"),
    (868964, 450, 524, "morikawa_mihono", "sasorigatame"),
    (868964, 525, 611, "akiyoshi_nekoko", "sasorigatame"),
    # Zettai Saikyou ☆ Oppai Sensou!!
    (536888, 348, 448, "kotone (zettai saikyou)", "annie"),
    (1481653, 352, 452, "kotone (zettai saikyou)", "annie"),
    # Amatarasu Riddle Star -
    (1033787, 2, 234, "ai_(amatarasu_riddle_star)", "syroh"),
    (1033787, 272, 540, "kokoro_judit", "2-g"),
    (1033787, 541, 731, "hatta_madori", "syroh"),
    (1033787, 776, 1118, "yukishiro miu", "annie"),
    (1033787, 1119, 1202, "arisu_rina", "annie"),
    (1033787, 1203, 1422, "arisu_yua", "2-g"),

    (1499212, 3, 235, "ai_(amatarasu_riddle_star)", "syroh"),
    (1499212, 273, 541, "kokoro_judit", "2-g"),
    (1499212, 542, 732, "hatta_madori", "syroh"),
    (1499212, 777, 1119, "yukishiro miu", "annie"),
    (1499212, 1120, 1202, "arisu_rina", "annie"),
    (1499212, 1203, 1421, "arisu_yua", "2-g"),

    (1115664, 1327, 1527, "ai_(amatarasu_riddle_star)", "syroh"),
    (1115664, 1528, 1666, "hatta_madori", "syroh"),
    (1115664, 805, 1326, "yukishiro miu", "annie"),
    (1115664, 1744, 1775, "arisu_rina", "annie"),
    (1115664, 1, 543, "arisu_yua", "2-g"),
    (1115664, 544, 804, "kokoro_judit", "2-g"),
    # Zettai Junshu New Kozukuri World
    (1008830, 361, 440, "tadokoro_minami", "2-g"),
    (1008830, 441, 530, "kasugai_noa", "2-g"),
    # Zettai Seifuku 
    (799866, 24, 149, "urushino_himeko", "sukoyaka gyuunyuu"),
    (799866, 758, 933, "uwaba_shiera", "araiguma"),
    # Yuuwaku Scramble
    (970288, 3, 302, "hoshimi_asuka", "hinata nao"),
    (970288, 303, 593, "hoshimi_yuki", "hinata nao"),
    (970288, 594, 872, "hoshimi_miku", "hinata nao"),

    (3588537, 3, 301, "hoshimi_asuka", "hinata nao"),
    (3588537, 302, 592, "hoshimi_yuki", "hinata nao"),
    (3588537, 593, 871, "hoshimi_miku", "hinata nao"),

    (3467981, 1, 456, "hoshimi_asuka", "hinata nao"),
    (3467981, 457, 888, "hoshimi_yuki", "hinata nao"),
    (3467981, 889, 1448, "hoshimi_miku", "hinata nao"),
    # Ojou-sama to Himitsu no Otome
    (827748, 56, 244, "himekouji_haruka", "sakurazaka tsuchiyu"),
    (827748, 245, 474, "kamimori_airi", "hinata nao"),
    (827748, 475, 683, "shirayuki_yumina", "sakurazaka tsuchiyu"),
    (827748, 684, 907, "saionji_saya", "hinata nao"),

    (2263122, 2, 189, "himekouji_haruka", "sakurazaka tsuchiyu"),
    (2263122, 190, 418, "kamimori_airi", "hinata nao"),
    (2263122, 419, 626, "shirayuki_yumina", "sakurazaka tsuchiyu"),
    (2263122, 627, 849, "saionji_saya", "hinata nao"),

    (1578575, 1, 817, "himekouji_haruka", "sakurazaka tsuchiyu"),
    (1578575, 818, 1513, "kamimori_airi", "hinata nao"),
    (1578575, 1514, 2000, "shirayuki_yumina", "sakurazaka tsuchiyu"),
    (1578589, 1, 350, "shirayuki_yumina", "sakurazaka tsuchiyu"),
    (1578589, 351, 1070, "saionji_saya", "hinata nao"),
    # Imouto Paradise! 2
    (600053, 2, 200, "nanase momoka", "itou life"),
    (600053, 201, 360, "nanase ririna", "itou life"),
    (600053, 361, 490, "nanase yuzu", "itou life"),
    (600053, 491, 671, "nanase chiharu", "itou life"),
    (600053, 672, 880, "nanase shizuku", "itou life"),
    (600053, 1118, 1150, "nanase momoka", "itou life"),
    (600053, 1151, 1183, "nanase ririna", "itou life"),
    (600053, 1184, 1228, "nanase yuzu", "itou life"),
    (600053, 1229, 1257, "nanase chiharu", "itou life"),
    (600053, 1258, 1306, "nanase shizuku", "itou life"),
    (1158457, 2, 259, "nanase momoka", "itou life"),
    (1158457, 260, 465, "nanase ririna", "itou life"),
    (1158457, 466, 625, "nanase yuzu", "itou life"),
    (1158457, 626, 878, "nanase chiharu", "itou life"),
    (1158457, 879, 1159, "nanase shizuku", "itou life"),
    (1977150, 13, 1452, "nanase momoka", "itou life"),
    (1977150, 1453, 2000, "nanase ririna", "itou life"),
    (1977160, 1, 892, "nanase ririna", "itou life"),
    (1977160, 893, 2000, "nanase yuzu", "itou life"),
    (1977178, 1, 1136, "nanase yuzu", "itou life"),
    (1977178, 1137, 2000, "nanase chiharu", "itou life"),
    (1977187, 1, 396, "nanase chiharu", "itou life"),
    (1977187, 397, 2000, "nanase shizuku", "itou life"),
    (1977201, 1, 600, "nanase shizuku", "itou life"),
    # Imouto Paradise
    (332150, 3, 130, "nanase_aya", "itou life"),
    (332150, 131, 304, "nanase_rio", "itou life"),
    (332150, 305, 428, "nanase_hiyori", "itou life"),
    (332150, 429, 550, "nanase_koharu", "itou life"),
    (332150, 551, 694, "nanase_michika", "itou life"),

    (580135, 2, 98, "nanase_aya", "itou life"),
    (580135, 99, 233, "nanase_rio", "itou life"),
    (580135, 234, 320, "nanase_hiyori", "itou life"),
    (580135, 321, 420, "nanase_koharu", "itou life"),
    (580135, 421, 542, "nanase_michika", "itou life"),

    (731862, 2, 98, "nanase_aya", "itou life"),
    (731862, 99, 234, "nanase_rio", "itou life"),
    (731862, 235, 322, "nanase_hiyori", "itou life"),
    (731862, 323, 423, "nanase_koharu", "itou life"),
    (731862, 424, 546, "nanase_michika", "itou life"),

    (1438033, 2, 98, "nanase_aya", "itou life"),
    (1438033, 99, 233, "nanase_rio", "itou life"),
    (1438033, 234, 320, "nanase_hiyori", "itou life"),
    (1438033, 321, 420, "nanase_koharu", "itou life"),
    (1438033, 421, 542, "nanase_michika", "itou life"),

    (1977135, 8, 187, "nanase_aya", "itou life"),
    (1977135, 188, 379, "nanase_rio", "itou life"),
    (1977135, 380, 499, "nanase_hiyori", "itou life"),
    (1977135, 500, 655, "nanase_koharu", "itou life"),
    (1977135, 656, 785, "nanase_michika", "itou life"),
    # Demon Busters
    (756187, 51, 243, "shizu_konoka", "itou life"),
    (756187, 244, 439, "lizera_(demon_busters)", "itou life"),
    (756187, 440, 653, "nakahime_karen", "itou life"),
    (756187, 654, 849, "hinata_ai", "itou life"),

    (3467987, 1, 741, "shizu_konoka", "itou life"),
    (3467987, 742, 1501, "lizera_(demon_busters)", "itou life"),
    (3467988, 1, 656, "nakahime_karen", "itou life"),
    (3467988, 657, 1336, "hinata_ai", "itou life"),
    # Icha Puri!
    (618693, 2, 121, "tenjouin_sakura", "itou life"),
    (618693, 122, 245, "tenjouin_yuzuna", "itou life"),
    (618693, 246, 361, "tenjouin_koyuri", "itou life"),
    (618693, 362, 481, "wakatsuki_sumireko", "itou life"),
    (618693, 482, 549, "kagurazaka_tsubaki", "saeki nao"),
    (618693, 550, 567, "misaki_futaba", "saeki nao"),

    (1577922, 5, 256, "tenjouin_sakura", "itou life"),
    (1577922, 257, 592, "tenjouin_yuzuna", "itou life"),
    (1577922, 593, 856, "tenjouin_koyuri", "itou life"),
    (1577922, 857, 1054, "wakatsuki_sumireko", "itou life"),
    (1577922, 1055, 1270, "kagurazaka_tsubaki", "saeki nao"),
    (1577922, 1271, 1414, "misaki_futaba", "saeki nao"),
    # Amanatsu Adolescence
    (1043759, 2, 22, "hyuuga_natsu", "hisama kumako"),
    (1043759, 23, 32, "sasha_mayakovskaya", "hitsuji takako"),
    (1043759, 88, 161, "hyuuga_natsu", "hisama kumako"),
    (1043759, 162, 230, "sasha_mayakovskaya", "hitsuji takako"),
    (1043759, 231, 301, "suzakano_ryou", "hisama kumako"),
    (1043759, 302, 380, "amakasu_amane", "hisama kumako"),
    # Shougun-sama wa Otoshigoro
    (1175803, 10, 121, "tokugawa_muneharu", "shona mitsuishi"),
    (1175803, 122, 223, "rin_(shougun-sama_wa_otoshigoro)", "kaniya shiku"),
    (1175803, 224, 328, "mitsui_tamaki", "kaniya shiku"),
    (1175803, 329, 478, "tokuda_yoshimune", "shona mitsuishi"),

    (1389173, 15, 89, "tokugawa_mitsukuni", "masaki kei"),
    (1389173, 92, 134, "tokugawa_muneharu", "shona mitsuishi"),
    (1389173, 138, 163, "rin_(shougun-sama_wa_otoshigoro)", "kaniya shiku"),
    (1389173, 167, 202, "mitsui_tamaki", "kaniya shiku"),
    (1389173, 215, 260, "tokuda_yoshimune", "shona mitsuishi"),

    (3687915, 1, 238, "tokugawa_muneharu", "shona mitsuishi"),
    (3687915, 239, 504, "rin_(shougun-sama_wa_otoshigoro)", "kaniya shiku"),
    (3687915, 536, 733, "mitsui_tamaki", "kaniya shiku"),
    (3687915, 766, 1260, "tokuda_yoshimune", "shona mitsuishi"),

    (3687916, 1, 288, "tokugawa_mitsukuni", "masaki kei"),
    (3687916, 289, 322, "tokugawa_muneharu", "shona mitsuishi"),
    (3687916, 323, 360, "rin_(shougun-sama_wa_otoshigoro)", "kaniya shiku"),
    (3687916, 361, 414, "mitsui_tamaki", "kaniya shiku"),
    (3687916, 415, 450, "tokuda_yoshimune", "shona mitsuishi"),
    # sousaku kanojo no renai koushiki
    (3425319, 1, 432, "ayase_aisa", None),
    (3425319, 433, 1197, "tsukimizaka_kiriha", None),
    (3425319, 1198, 1492, "nagima_yumemi", None),
    (3425319, 1493, 1745, "yukisaki_erena", None),
    (2351500, 2, 10, "ayase_aisa", None),
    (2351500, 42, 81, "ayase_aisa", None),
    (2351500, 11, 20, "tsukimizaka_kiriha", None),
    (2351500, 82, 107, "tsukimizaka_kiriha", None),
    (2351500, 108, 147, "nagima_yumemi", None),
    (2351500, 148, 173, "yukisaki_erena", None),
    (2070784, 19, 36, "ayase_aisa", None),
    (2070784, 51, 53, "ayase_aisa", None),
    (2070784, 132, 147, "yukisaki_erena", None),
    (2070784, 185, 210, "ayase_aisa", None),
    (2070784, 211, 236, "yukisaki_erena", None),
    (2070784, 253, 428, "nagima_yumemi", None),
    (2070784, 604, 744, "yukisaki_erena", None),
    (2070784, 762, 800, "ayase_aisa", None),
    (2070784, 815, 956, "ayase_aisa", None),
    (2070784, 54, 102, "tsukimizaka_kiriha", None),
    (2070784, 429, 603, "tsukimizaka_kiriha", None),
    # Shukufuku no Kanenone wa, Sakurairo no Kaze Totomoni
    (1321028, 2, 112, "ootori_maria", "anapom"),
    (1321028, 113, 229, "nishikujou_kanon", "anapom"),
    (1321028, 230, 362, "kitazono_saya", "anapom"),
    (1321028, 363, 489, "shinonome_urara", "anapom"),
    # Kokoro ga Tsunagu Koi Shirube
    (1322592, 5, 350, "kujou_himeno", None),
    (1322592, 351, 643, "nanase_hinata_(kokoro_ga_tsunagu_koi_shirube)", None),
    (1322592, 644, 963, "kokoro_(kokoro_ga_tsunagu_koi_shirube)", None),
    (1322592, 964, 1320, "tsukihoshi_mashiro", None),
    # God Sisters
    (1217312, 4, 440, "konishi_ami", "kakao"),
    (1217312, 441, 471, "konishi_ria, konishi_yuu", "amamine"),
    (1217312, 472, 512, "konishi_ami, konishi_riko", "kakao"),
    (1217312, 513, 963, "konishi_ria", "amamine"),
    (1217312, 964, 1423, "konishi_riko", "kakao"),
    (1217312, 1424, 1906, "konishi_yuu", "amamine"),
    # Dolphin Blade
    (1122200, 3, 104, "asuka_noa", "konata"),
    (1122200, 105, 276, None, "mikeou"),
    (1122200, 277, 392, "ui_sengouji", "amamine"),
    (1122200, 393, 494, "otoha_shinogiri", "hinata momo"),
    (1122200, 495, 651, None, "kukiha"),
    # Koi wa Yumemiru Mouretsu Girl!
    (1009125, 7, 384, "mioka_aoi", "naenae"),
    (1009125, 385, 669, "koshiro_erika", "naenae"),
    (1009125, 673, 1060, "yuunagi_shizuku", "naenae"),
    (1009125, 1061, 1331, "shiosaki_yuki", "amamine"),
    (1009125, 1332, 1349, None, "niki"),
    (1009125, 1350, 1368, "chie", "niki"),
    (1257428, 1, 29, None, "niki"),
    (1257428, 30, 129, "mioka_aoi", "naenae"),
    (1257428, 130, 163, "chie", "niki"),
    (1257428, 164, 256, "koshiro_erika", "naenae"),
    (1257428, 257, 353, "yuunagi_shizuku", "naenae"),
    (1257428, 354, 453, "shiosaki_yuki", "amamine"),
    # Garudoma
    (2653109, 3, 115, "fuyusaki_aiko", None),
    (2653109, 197, 259, "natsumi_rin", None),
    (2653109, 371, 437, "akishima_hiyori", None),
    (2653109, 438, 550, "fuyusaki_aiko", None),
    (2653109, 632, 694, "natsumi_rin", None),
    (2653109, 806, 872, "akishima_hiyori", None),
    (2653109, 873, 987, "fuyusaki_aiko", None),
    (2653109, 1069, 1131, "natsumi_rin", None),
    (2653109, 1243, 1309, "akishima_hiyori", None),
    (2653109, 1310, 1378, "fuyusaki_aiko", None),
    (2653109, 1536, 1598, "natsumi_rin", None),
    (2653109, 1710, 1776, "akishima_hiyori", None),
    (2836574, 3, 236, "fuyusaki_aiko", None),
    (2836574, 673, 1271, "fuyusaki_aiko, natsumi_rin, akishima_hiyori", None),
    (2836574, 1272, 1549, "natsumi_rin", None),
    (2836574, 1726, 2000, "akishima_hiyori", None),

    (3442547, 1, 106, "fuyusaki_aiko", None),
    (3442547, 107, 206, "natsumi_rin", None),
    (3442547, 207, 603, "akishima_hiyori", None),
    (3442546, 1, 2000, "akishima_hiyori", None),
    (3442545, 1, 780, "fuyusaki_aiko", None),
    (3442545, 781, 2000, "natsumi_rin", None),
    # Hatsukoi 1/1
    (504063, 1070, 2000, "makabe_midori", "koizumi amane"),
    (504064, 2, 97, "makabe_midori", "koizumi amane"),
    # Natsuzora no Perseus
    (550675, 1, 44, "tohno_ren", "shona mitsuishi"),
    (550675, 45, 168, "sawatari_tohka", "shona mitsuishi"),
    (634944, 2, 10, "sawatari_tohka", "shona mitsuishi"),
    (634944, 11, 285, "tohno_ren", "shona mitsuishi"),
    (634944, 286, 617, "minakawa_sui", "takasaki maco"),
    (634944, 618, 1045, "hishida_ayame", "yuzuna hiyo"),
    (634944, 1046, 1121, "tohno_ren", "shona mitsuishi"),
    (634944, 1122, 1239, "sawatari_tohka", "shona mitsuishi"),
    # Hanikami CLOVER
    (899587, 4, 14, "saeki_rio", "kakao"),
    (899587, 68, 419, "saeki_rio", "kakao"),
    (899587, 39, 54, "suoh_emiru", "kakao"),
    (899587, 773, 1214, "suoh_emiru", "kakao"),
    (1442740, 1, 32, "saeki_rio", "kakao"),
    (1442740, 33, 70, "suoh_emiru", "kakao"),
    (1442740, 144, 179, "saeki_rio", "kakao"),
    (1442740, 180, 224, "suoh_emiru", "kakao"),

    (1442854, 297, 568, "saeki_rio", "kakao"),
    (1442820, 3, 362, "suoh_emiru", "kakao"),
    # Ama Koi Syrups
    (790794, 2, 181, "watanuki_tsuyuri", "pan"),
    (790794, 182, 362, "lawes_anastesia", "suimya"),
    (790794, 363, 525, "katori_rin", "suimya"),
    (790794, 526, 699, "kusaka_hozumi", "pan"),

    (1166516, 2, 181, "watanuki_tsuyuri", "pan"),
    (1166516, 182, 363, "lawes_anastesia", "suimya"),
    (1166516, 364, 526, "katori_rin", "suimya"),
    (1166516, 527, 700, "kusaka_hozumi", "pan"),

    (1166649, 2, 153, "watanuki_tsuyuri", "pan"),
    (1166649, 154, 313, "lawes_anastesia", "suimya"),
    (1166649, 314, 475, "katori_rin", "suimya"),
    (1166649, 447, 598, "kusaka_hozumi", "pan"),
    # Tenshi☆Souzou RE-BOOT!
    (2536708, 3, 466, "shirayuki_noa", "kobuichi"),
    (2536708, 467, 708, "ozato_fumika", "hadumi rio"),
    (2537215, 4, 409, "tanikaze_amane", "muririn"),
    (2537215, 410, 736, "kohibari_kurumi", "muririn"),
    (2537215, 737, 1214, "hoshikawa_kaguya", "kobuichi"),
    (2537215, 1215, 1510, "takadate_orie", "hokkana"),
    (3423289, 1, 2000, "shirayuki_noa", "kobuichi"),
    (3423288, 1, 2000, "shirayuki_noa", "kobuichi"),
    (3423291, 1, 2000, "shirayuki_noa", "kobuichi"),
    (3423290, 1, 2000, "shirayuki_noa", "kobuichi"),
    (3422989, 1, 2000, "ozato_fumika", "hadumi rio"),
    (3422985, 1, 2000, "kohibari_kurumi", "muririn"),
    (3422986, 1, 2000, "kohibari_kurumi", "muririn"),
    (3423069, 1, 2000, "hoshikawa_kaguya", "kobuichi"),
    (3423070, 1, 2000, "hoshikawa_kaguya", "kobuichi"),
    (3423071, 1, 2000, "hoshikawa_kaguya", "kobuichi"),
    (3423072, 1, 2000, "hoshikawa_kaguya", "kobuichi"),
    (3421521, 1, 2000, "tanikaze_amane", "muririn"),
    (3421522, 1, 2000, "tanikaze_amane", "muririn"),
    (3422994, 1, 2000, "takadate_orie", "hokkana"),
    (3422993, 1, 2000, "takadate_orie", "hokkana"),
    # Limelight Lemonade Jam
    (3556090, 1, 2000, "shimakoshi_tsukimi", "hokkana"),
    (3590315, 1, 2000, "shimakoshi_tsukimi", "hokkana"),
    (3590316, 1, 2000, "shimakoshi_tsukimi", "hokkana"),
    (3590317, 1, 2000, "shimakoshi_tsukimi", "hokkana"),
    (3556158, 1, 2000, "koishikawa_miku", "kobuichi"),
    (3556159, 1, 2000, "koishikawa_miku", "kobuichi"),
    (3556077, 1, 2000, "harumi_ena", "muririn"),
    (3556078, 1, 2000, "harumi_ena", "muririn"),
    (3556079, 1, 2000, "harumi_ena", "muririn"),
    (3556080, 1, 2000, "harumi_ena", "muririn"),
    (3556081, 1, 2000, "harumi_ena", "muririn"),
    (3556082, 1, 2000, "harumi_ena", "muririn"),
    (3556083, 1, 2000, "harumi_ena", "muririn"),
    (3556084, 1, 2000, "harumi_ena", "muririn"),
    (3556085, 1, 2000, "nabari_anju", "kobuichi"),
    (3556086, 1, 2000, "nabari_anju", "kobuichi"),
    (3556088, 1, 2000, "nabari_anju", "kobuichi"),
    (3556089, 1, 2000, "nabari_anju", "kobuichi"),
    (3556156, 1, 2000, "saen_nayuka", "hadumi rio"),
    (3556157, 1, 2000, "saen_nayuka", "hadumi rio"),
    (3556094, 1, 2000, "futamihara_ririko", "muririn"),
    (3556095, 1, 2000, "futamihara_ririko", "muririn"),
    (3556096, 1, 2000, "futamihara_ririko", "muririn"),
    (3556097, 1, 2000, "futamihara_ririko", "muririn"),
    (3553799, 22, 62, "harumi_ena", "muririn"),
    (3553799, 63, 87, "nabari_anju", "kobuichi"),
    (3553799, 88, 129, "shimakoshi_tsukimi", "hokkana"),
    (3553799, 130, 174, "futamihara_ririko", "muririn"),
    (3553799, 175, 189, "koishikawa_miku", "kobuichi"),
    (3553799, 190, 208, "saen_nayuka", "hadumi rio"),

    (3638988, 3, 1008, "harumi_ena", "muririn"),
    (3638988, 1009, 1466, "nabari_anju", "kobuichi"),
    (3590156, 2, 604, "shimakoshi_tsukimi", "hokkana"),
    (3590156, 605, 1321, "futamihara_ririko", "muririn"),
    (3590156, 1322, 1581, "koishikawa_miku", "kobuichi"),
    (3638988, 1467, 1890, "saen_nayuka", "hadumi rio"),

    # cafe stella to shinigami no chou
    (1538399, 1, 2000, "akizuki_kanna", "kobuichi"),
    (1538355, 1, 2000, "shiki_natsume", "muririn"),
    (1538430, 1, 2000, "sumizome_nozomi", "muririn"),
    (1538439, 1, 2000, "hiuchidani_mei", "kobuichi"),
    (1538498, 1, 1044, "shioyama_suzune", "muririn"),

    (1522825, 3, 145, "akizuki_kanna", "kobuichi"),
    (1522825, 146, 223, "shiki_natsume", "muririn"),
    (1522825, 224, 265, "sumizome_nozomi", "muririn"),
    (1522825, 266, 312, "hiuchidani_mei", "kobuichi"),
    (1522825, 313, 382, "shioyama_suzune", "muririn"),

    (1536430, 12, 512, "akizuki_kanna", "kobuichi"),
    (1536430, 513, 1134, "shiki_natsume", "muririn"),
    (1536430, 1135, 1778, "sumizome_nozomi", "muririn"),
    (1536431, 1, 513, "hiuchidani_mei", "kobuichi"),
    (1536431, 514, 1043, "shioyama_suzune", "muririn"),

    # Sengokuhime 5
    (809507, 171, 210, "oda_nobuyuki_(sengoku_hime)", None),
    # amayui castle meister
    (1067242, 72, 341, "fia_(amayui_castle_meister)", "yano mitsuki"),
    (1117997, 7, 112, "fia_(amayui_castle_meister)", "yano mitsuki"),
    (1179437, 16, 35, "fia_(amayui_castle_meister)", "yano mitsuki"),
    # secret love
    (2990687, 4, 423, "sawa_chiaki", "k-ko"),
    (2999687, 424, 776, "akatsuka_haru", "k-ko"),
    (2999687, 777, 1177, "momouchi_kaede", "mango pudding"),
    (2999687, 1178, 1587, "natori_misa", "mango pudding"),
    (3328435, 2, 161, "sawa_chiaki", "k-ko"),
    (3328435, 170, 333, "akatsuka_haru", "k-ko"),
    (3328435, 334, 475, "momouchi_kaede", "mango pudding"),
    (3328435, 477, 652, "natori_misa", "mango pudding"),
    # IxSHE Tell
    (1189877, 9, 161, "yuuki_ayaka", None),
    (1189877, 179, 394, "koshimizu_kasumi", None),
    (1189877, 395, 592, "hanamori_shiori", None),
    (1189877, 593, 768, "yamabuki_yoshino", None),
    (1189877, 769, 955, "koduka_yui", None),

    (1263294, 3, 166, "yuuki_ayaka", None),
    (1263294, 167, 308, "yamabuki_yoshino", None),

    (1990347, 2, 198, "hanamori_shiori", None),
    (1990347, 200, 415, "koshimizu_kasumi", None),
    (1990347, 417, 568, "yuuki_ayaka", None),
    (1990347, 570, 756, "koduka_yui", None),
    (1990347, 758, 933, "yamabuki_yoshino", None),


    (2566350, 7, 159, "yuuki_ayaka", None),
    (2566350, 177, 392, "koshimizu_kasumi", None),
    (2566350, 393, 590, "hanamori_shiori", None),
    (2566350, 591, 766, "yamabuki_yoshino", None),
    (2566350, 768, 953, "koduka_yui", None),
    (2566350, 954, 1022, "yuuki_ayaka", None),
    (2566350, 1031, 1109, "koshimizu_kasumi", None),
    (2566350, 1110, 1171, "hanamori_shiori", None),
    (2566350, 1172, 1237, "yamabuki_yoshino", None),
    (2566350, 1238, 1307, "koduka_yui", None),
    # Houkago Cinderella
    (1719085, 7, 170, "oujou_maria", "rinks"),
    (1719085, 171, 313, "tayori_tanomi", "rinks"),
    (1719085, 314, 500, "tsukushima_tsukushi", "rinks"),
    (1719085, 501, 663, "osanami_youka", "rinks"),
    (1719085, 664, 820, "usagawa_yukiko", "rinks"),

    (2127509, 2, 31, "oujou_maria", "rinks"),
    (2127509, 32, 73, "tayori_tanomi", "rinks"),
    (2127509, 74, 107, "tsukushima_tsukushi", "rinks"),
    (2127509, 108, 145, "osanami_youka", "rinks"),
    (2127509, 146, 176, "usagawa_yukiko", "rinks"),
    (2127509, 177, 225, "oujou_maria", "rinks"),
    (2127509, 226, 282, "tayori_tanomi", "rinks"),
    (2127509, 283, 351, "tsukushima_tsukushi", "rinks"),
    (2127509, 352, 409, "osanami_youka", "rinks"),
    (2127509, 410, 454, "usagawa_yukiko", "rinks"),

    (2339211, 2, 163, "oze_hazuki", "rinks"),
    (2339211, 164, 471, "kurumai_mikuru", "rinks"),
    (2339211, 472, 546, "amanogawa_milky", "rinks"),
    (2339211, 547, 744, "watarase_nene", "rinks"),
    (2339211, 745, 917, "madokawa_serika", "rinks"),
    (2339211, 918, 1069, "natsugoe_chiho", "rinks"),
    (2339211, 1070, 1124, "oze_hazuki", "rinks"),
    (2339211, 1125, 1202, "kurumai_mikuru", "rinks"),
    (2339211, 1203, 1220, "amanogawa_milky", "rinks"),
    (2339211, 1221, 1344, "watarase_nene", "rinks"),
    (2339211, 1345, 1391, "madokawa_serika", "rinks"),
    (2339211, 1392, 1496, "natsugoe_chiho", "rinks"),

    (2595337, 2, 117, "oze_hazuki", "rinks"),
    (2595337, 118, 223, "kurumai_mikuru", "rinks"),
    (2595337, 225, 362, "watarase_nene", "rinks"),
    (2595337, 364, 473, "madokawa_serika", "rinks"),
    (2595337, 474, 571, "natsugoe_chiho", "rinks"),
    # FLIP＊FLOP
    (2362035, 1, 2000, "io_(flip_flop)", None),
    (2475523, 4, 248, "tsukigaoka_ran", None),
    # Pure x Connect
    (820343, 14, 184, "konno_arisa", 'ameto yuki'),
    (820343, 185, 342, "shinozaki_ayumi_(pure_x_connect)", "ameto yuki"),
    (820343, 343, 547, "doumyouji_moemi", "ameto yuki"),
    (820343, 548, 726, "makihara_shiho_(pure_x_connect)", "ameto yuki"),
    (820343, 727, 909, "ichinose_sara", "ameto yuki"),
    (820343, 910, 933, "konno_arisa", 'ameto yuki'),
    (820343, 934, 945, "shinozaki_ayumi_(pure_x_connect)", "ameto yuki"),
    (820343, 946, 954, "doumyouji_moemi", "ameto yuki"),
    (820343, 955, 967, "makihara_shiho_(pure_x_connect)", "ameto yuki"),
    (820343, 968, 990, "ichinose_sara", "ameto yuki"),

    (2100756, 5, 175, "konno_arisa", 'ameto yuki'),
    (2100756, 176, 333, "shinozaki_ayumi_(pure_x_connect)", "ameto yuki"),
    (2100756, 334, 538, "doumyouji_moemi", "ameto yuki"),
    (2100756, 539, 717, "makihara_shiho_(pure_x_connect)", "ameto yuki"),
    (2100756, 718, 900, "ichinose_sara", "ameto yuki"),
    (2100756, 901, 1032, "konno_arisa", 'ameto yuki'),
    (2100756, 1033, 1138, "shinozaki_ayumi_(pure_x_connect)", "ameto yuki"),
    (2100756, 1139, 1284, "doumyouji_moemi", "ameto yuki"),
    (2100756, 1285, 1413, "makihara_shiho_(pure_x_connect)", "ameto yuki"),
    (2100756, 1414, 1545, "ichinose_sara", "ameto yuki"),

    (2100756, 1546, 1551, "konno_arisa", 'ameto yuki'),
    (2100756, 1552, 1562, "shinozaki_ayumi_(pure_x_connect)", "ameto yuki"),
    (2100756, 1563, 1568, "doumyouji_moemi", "ameto yuki"),
    (2100756, 1569, 1577, "makihara_shiho_(pure_x_connect)", "ameto yuki"),
    (2100756, 1578, 1581, "ichinose_sara", "ameto yuki"),
    # DRACU-RIOT!
    (875699, 1, 736, "yarai_miu", "muririn"),
    (875699, 737, 1746, "mera_azusa", "muririn"),
    (875699, 1747, 2000, "inamura_rio", "kobuichi"),
    (875672, 1, 295, "inamura_rio", "kobuichi"),
    (875672, 296, 1031, "elena_olegovna_owen", "kobuichi"),
    (875672, 1032, 1335, "nicola_cepheus", "muririn"),
    # Senren*banka
    (3442432, 1, 2000, "tomotake_yoshino", "kobuichi"),
    (1890822, 1, 994, "tomotake_yoshino", "kobuichi"),
    (960624, 3, 646, "tomotake_yoshino", "kobuichi"),
    (960624, 647, 1018, "hitachi_mako", "muririn"),
    (960624, 1019, 1256, "murasame_(senren)", "muririn"),
    (960701, 1, 242, "lena_liechtenauer", "kobuichi"),
    (960701, 243, 396, "kurama_koharu", "senji"),
    (960701, 397, 557, "maniwa_roka", "muririn"),

    (1890779, 7, 726, "murasame_(senren)", "muririn"),
    (1890779, 811, 1504, "lena_liechtenauer", "kobuichi"),
    (1890779, 1505, 1924, "kurama_koharu", "senji"),
    (1890779, 1940, 2000, "maniwa_roka", "muririn"),
    (1890811, 1, 305, "maniwa_roka", "muririn"),
    (1890811, 418, 2000, "tomotake_yoshino", "kobuichi"),
    (1890822, 1, 993, "tomotake_yoshino", "kobuichi"),
    (1890822, 994, 2000, "hitachi_mako", "muririn"),

    (1891159, 111, 1356, "tomotake_yoshino", "kobuichi"),
    (1891159, 1357, 1875, "hitachi_mako", "muririn"),
    (1891187, 1, 255, "hitachi_mako", "muririn"),
    (1891187, 256, 665, "murasame_(senren)", "muririn"),
    (1891187, 666, 1055, "lena_liechtenauer", "kobuichi"),
    (1891187, 1056, 1355, "kurama_koharu", "senji"),
    (1891187, 1356, 1655, "maniwa_roka", "muririn"),
    # Sanoba Witch
    (3424478, 1, 2000, "ayachi_nene", "muririn"),
    (3424479, 1, 2000, "ayachi_nene", "muririn"),
    (3424480, 1, 2000, "ayachi_nene", "muririn"),
    (3424414, 1, 2000, "togakushi_touko", "kobuichi"),
    (3424415, 1, 2000, "togakushi_touko", "kobuichi"),
    (798685, 2, 407, "shiiba_tsumugi", "kobuichi"),
    (798685, 408, 958, "togakushi_touko", "kobuichi"),
    (798685, 959, 1338, "kariya_wakana", "kobuichi"),
    (798679, 3, 934, "ayachi_nene", "muririn"),
    (798679, 935, 1580, "inaba_meguru", "muririn"),
    (3424416, 1, 2000, "inaba_meguru", "muririn"),
    (3424417, 1, 2000, "inaba_meguru", "muririn"),
    (3424422, 1, 2000, "shiiba_tsumugi", "kobuichi"),
    (3424423, 1, 2000, "shiiba_tsumugi", "kobuichi"),
    (3424413, 1, 2000, "kariya_wakana", "kobuichi"),
    (2619777, 3, 428, "ayachi_nene", "muririn"),
    (2619777, 429, 756, "inaba_meguru", "muririn"),
    (2619777, 757, 950, "shiiba_tsumugi", "kobuichi"),
    (2619777, 951, 1255, "togakushi_touko", "kobuichi"),
    (2619777, 1256, 1433, "kariya_wakana", "kobuichi"),
    # RIDDLE JOKER
    (2984952, 1, 14, "arihara_nanami", "kobuichi"),
    (1541162, 1, 2000, "mitsukasa_ayase", "muririn"),
    (1543784, 1, 2000, "arihara_nanami", "kobuichi"),
    (1543991, 1, 2000, "shikibe_mayu", "muririn"),
    (1544108, 1, 2000, "nijouin_hazuki", "kobuichi"),
    (1544147, 1, 931, "mibu_chisaki", "kobuichi"),
    (1468670, 1, 972, "mitsukasa_ayase", "muririn"),
    (1468670, 973, 1815, "arihara_nanami", "kobuichi"),
    (1468670, 1816, 2000, "shikibe_mayu", "muririn"),
    (1468698, 1, 474, "shikibe_mayu", "muririn"),
    (1468698, 475, 976, "nijouin_hazuki", "kobuichi"),
    (1468698, 977, 1365, "mibu_chisaki", "kobuichi"),

    (1204840, 3, 492, "mitsukasa_ayase", "muririn"),
    (1204840, 493, 930, "arihara_nanami", "kobuichi"),
    (1204840, 931, 1297, "shikibe_mayu", "muririn"),
    (1204840, 1298, 1659, "nijouin_hazuki", "kobuichi"),
    (1204840, 1660, 1908, "mibu_chisaki", "kobuichi"),
    (1204840, 1917, 1925, "arihara_nanami, shikibe_mayu", "muririn, kobuichi"),
    (1204840, 1937, 1954, "arihara_nanami, mibu_chisaki", "kobuichi"),
    # Amairo IsleNauts
    (607261, 2, 31, "shirley_warwick", "kobuichi"),
    (607261, 32, 63, "amagiri_yune", "muririn"),
    (607261, 64, 99, "shiraga_airi", "kobuichi"),
    (607261, 100, 125, "masaki_gaillard", "muririn"),
    (607261, 126, 155, "hinomiya_konoka", "kobuichi"),
    (607261, 156, 187, "tia_hohenwerfen", "muririn"),

    (614344, 2, 233, "shirley_warwick", "kobuichi"),
    (614344, 234, 505, "amagiri_yune", "muririn"),
    (614344, 506, 697, "shiraga_airi", "kobuichi"),
    (614344, 698, 840, "masaki_gaillard", "muririn"),
    (614344, 841, 957, "hinomiya_konoka", "kobuichi"),
    (614344, 958, 1111, "tia_hohenwerfen", "muririn"),

    (614339, 2, 533, "shirley_warwick", "kobuichi"),
    (614339, 534, 1213, "amagiri_yune", "muririn"),
    (614339, 1214, 1635, "shiraga_airi", "kobuichi"),
    (614339, 1636, 1982, "masaki_gaillard", "muririn"),
    (614339, 1983, 2000, "hinomiya_konoka", "kobuichi"),
    (614338, 2, 251, "hinomiya_konoka", "kobuichi"),
    (614338, 252, 846, "tia_hohenwerfen", "muririn"),

    (631813, 1, 532, "shirley_warwick", "kobuichi"),
    (631813, 533, 1203, "amagiri_yune", "muririn"),
    (631813, 1204, 1596, "shiraga_airi", "kobuichi"),
    (631813, 1597, 2000, "masaki_gaillard", "muririn"),
    (631857, 1, 36, "masaki_gaillard", "muririn"),
    (631857, 37, 300, "hinomiya_konoka", "kobuichi"),
    (631857, 301, 890, "tia_hohenwerfen", "muririn"),
    # Tenshin Ranman
    (875478, 3, 190, "unohana_no_sakuyahime", "muririn"),
    (875478, 191, 436, "rindou_ruri", "kobuichi"),
    (875478, 437, 672, "chitose_sana", "muririn"),
    (875478, 673, 1069, "yamabuki_aoi", "kobuichi"),
    # Noble Works
    (878712, 2, 467, "kanemoto_akari", "kobuichi"),
    (878712, 468, 826, "tsukiyama_sena", "muririn"),
    (878712, 827, 1159, "masamune_shizuru", "kobuichi"),
    (878712, 1160, 1375, "kunihiro_hinata", "muririn"),

    (672070, 2, 648, "kanemoto_akari", "kobuichi"),
    (672070, 1426, 1852, "tsukiyama_sena", "muririn"),
    (672070, 1853, 2000, "masamune_shizuru", "kobuichi"),
    (672275, 1, 246, "masamune_shizuru", "kobuichi"),
    (672275, 316, 706, "kunihiro_hinata", "muririn"),
    (672275, 786, 856, "kunihiro_hinata", "muririn"),
    # Southern Cross Love Song / Minamijuujisei Renka
    (743876, 4, 295, "fujina_kanori", None),
    (743876, 296, 514, "elise_rosenthal", None),
    (743876, 515, 719, "naraoka_mitsuki", None),
    (743876, 720, 1011, "hasami_miyako", None),
    (743876, 1012, 1323, "tsutsumi_sakuya", None),
    # Sorceress*Alive!
    (1354083, 3, 14, "akina_randal", "shona mitsuishi"),
    (1354083, 15, 32, "yuzuriha_serval", "hayakawa halui"),
    (1354083, 33, 43, "mia_welch", "sakura misaki"),
    (1354083, 44, 51, "azuria_newfield", "shona mitsuishi"),
    (1354083, 200, 282, "akina_randal", "shona mitsuishi"),
    (1354083, 283, 361, "azuria_newfield", "shona mitsuishi"),
    (1354083, 362, 419, "mia_welch", "sakura misaki"),
    (1354083, 420, 479, "yuzuriha_serval", "hayakawa halui"),

    (1444193, 1, 352, "akina_randal", "shona mitsuishi"),
    (1444193, 353, 700, "azuria_newfield", "shona mitsuishi"),
    (1444193, 810, 1289, "mia_welch", "sakura misaki"),
    (1444193, 1888, 2000, "yuzuriha_serval", "hayakawa halui"),
    (1444194, 1, 2000, "yuzuriha_serval", "hayakawa halui"),
    # Ren'ai, Hajimemashite
    (3255903, 1, 177, "tenshi-chan_(ren'ai_hajimemashite)", "fuyuichi monme"),
    (3255903, 180, 306, "aizawa_yukari", "unasaka"),
    (3255903, 308, 438, None, "yuunagi seshina"),
    (3255903, 439, 566, "inuya_komaru", "sacraneco"),
    (3255903, 567, 604, None, "unasaka"),
    (3255903, 605, 633, None, "fuyuichi monme"),
    (3554542, 2, 63, "tenshi-chan_(ren'ai_hajimemashite)", "fuyuichi monme"),
    (3554542, 64, 110, "aizawa_yukari", "unasaka"),
    (3554542, 111, 142, None, "yuunagi seshina"),
    (3554542, 143, 179, "inuya_komaru", "sacraneco"),
    (3554542, 258, 262, "tenshi-chan_(ren'ai_hajimemashite)", "fuyuichi monme"),
    (3554542, 263, 268, "aizawa_yukari", "unasaka"),
    (3554542, 269, 274, None, "yuunagi seshina"),
    (3554542, 275, 278, "inuya_komaru", "sacraneco"),
    # Koibana Ren'ai
    (2872360, 3, 40, "otome_kokoro", "yuuki rika"),
    (2872360, 42, 80, "adachi_chii", "yuunagi seshina"),
    (2872360, 81, 108, "harukaze_meguri", "fuyuichi monme"),
    (2872360, 109, 144, "yuugure_tokoyo", "fuyuichi monme"),
    (2872360, 145, 204, "koeda_fumi", "yuuki rika"),
    (2872360, 205, 259, "harukaze_inori", "yuuki rika"),
    (2872360, 260, 290, None, "yuunagi seshina"),

    (2692612, 2, 186, "otome_kokoro", "yuuki rika"),
    (2692612, 187, 332, "adachi_chii", "yuunagi seshina"),
    (2692612, 333, 459, "harukaze_meguri", "fuyuichi monme"),
    (2692612, 462, 575, "yuugure_tokoyo", "fuyuichi monme"),
    (2692612, 577, 621, None, "yuuki rika"),
    (2692612, 622, 637, None, "yuunagi seshina"),

    (2893453, 1, 2000, "adachi_chii", "yuunagi seshina"),
    (2893246, 1, 2000, "harukaze_meguri", "fuyuichi monme"),
    (2893243, 1, 2000, "otome_kokoro", "yuuki rika"),
    (2893244, 1, 792, "otome_kokoro", "yuuki rika"),
    (2893244, 793, 1656, "koeda_fumi", "yuuki rika"),
    (2893247, 1, 1242, "yuugure_tokoyo", "fuyuichi monme"),
    (2893247, 1243, 1536, None, "yuunagi seshina"),
    (2893247, 1537, 1704, None, "yuunagi seshina"),
    (2893249, 1, 930, "koigawara_mia", "yuunagi seshina"),
    (2891513, 1, 290, None, "yuuki rika"),
    (2891512, 1, 2000, None, "yuuki rika"),
    (2893248, 1, 800, "harukaze_inori", "yuuki rika"),
    (2893248, 801, 1415, "harukaze_inori", "yuuki rika"),
    # Futamata Ren'ai
    (3457068, 2, 149, "nobuta_yua", "fuyuichi monme"),
    (3457068, 153, 308, "toiro_kirame", "fuyuichi monme"),
    (3457068, 309, 449, "mikoshiba_rui", "yuunagi seshina"),
    (3457068, 450, 574, "umino_miyako", "yuuki rika"),

    (2412643, 4, 57, "nobuta_yua", "fuyuichi monme"),
    (2412643, 58, 128, "toiro_kirame", "fuyuichi monme"),

    (2205648, 2, 152, "nobuta_yua", "fuyuichi monme"),
    (2205648, 153, 308, "toiro_kirame", "fuyuichi monme"),
    (2205648, 309, 449, "mikoshiba_rui", "yuunagi seshina"),
    (2205648, 450, 574, "umino_miyako", "yuuki rika"),
    (2205648, 575, 605, None, "yuuki rika"),

    (2311617, 2, 78, "mikoshiba_rui", "yuunagi seshina"),
    (2311617, 79, 120, "umino_miyako", "yuuki rika"),

    (2891508, 1, 2000, "nobuta_yua", "fuyuichi monme"),
    (2891509, 1, 2000, "toiro_kirame", "fuyuichi monme"),
    (2893674, 1, 2000, "mikoshiba_rui", "yuunagi seshina"),
    (2891511, 1, 2000, "umino_miyako", "yuuki rika"),
    # Renai, Karichaimashita
    (1453395, 4, 189, "segawa_emi", "fuyuichi monme"),
    (1453395, 192, 344, "tenma_hasumi", "fuyuichi monme"),
    (1453395, 350, 493, None, "yuunagi seshina"),
    (1453395, 494, 646, "soraji_tsubaki", "yuuki rika"),
    (1453395, 647, 679, "mihama_saki", "yuuki rika"),
    (1453395, 680, 715, "kozeki_momoko", "yuuki rika"),
    (2043589, 3, 188, "segawa_emi", "fuyuichi monme"),
    (2043589, 191, 343, "tenma_hasumi", "fuyuichi monme"),
    (2043589, 349, 492, None, "yuunagi seshina"),
    (2043589, 493, 645, "soraji_tsubaki", "yuuki rika"),
    (2043589, 646, 678, "mihama_saki", "yuuki rika"),
    (2043589, 679, 714, "kozeki_momoko", "yuuki rika"),
    
    (1531217, 1, 626, "segawa_emi", "fuyuichi monme"),
    (1531217, 627, 1341, "tenma_hasumi", "fuyuichi monme"),
    (1531243, 1, 115, "tenma_hasumi", "fuyuichi monme"),
    (1531243, 116, 923, None, "yuunagi seshina"),
    (1531243, 924, 2000, "soraji_tsubaki", "yuuki rika"),
    (1531269, 1, 390, "soraji_tsubaki", "yuuki rika"),
    (1531269, 391, 560, "mihama_saki", "yuuki rika"),
    (1531269, 947, 1126, "kozeki_momoko", "yuuki rika"),

    (1562392, 4, 49, "segawa_emi", "fuyuichi monme"),
    (1562392, 50, 89, "tenma_hasumi", "fuyuichi monme"),
    (1562392, 90, 118, "segawa_emi, tenma_hasumi", "fuyuichi monme"),
    (1562392, 120, 120, "segawa_emi, tenma_hasumi", "fuyuichi monme"),
    # Renai x Royale
    (1786483, 6, 134, "hanamaru_mari", "fuyuichi monme"),
    (1786483, 140, 271, "harizome_shione", "fuyuichi monme"),
    (1786483, 273, 400, "komachi_nonoka", "yuunagi seshina"),
    (1786483, 401, 545, "amagamine_renna", "yuuki rika"),
    (1786483, 573, 590, "iyori_ao", "yuuki rika"),
    (1786483, 608, 671, "kagaya_yuna", "yuuki rika"),

    (1813072, 4, 654, "hanamaru_mari", "fuyuichi monme"),
    (1813072, 655, 1550, "harizome_shione", "fuyuichi monme"),
    (1813072, 1551, 2000, "komachi_nonoka", "yuunagi seshina"),
    (1813075,  1, 372, "komachi_nonoka", "yuunagi seshina"),
    (1813075, 373, 1470, "amagamine_renna", "yuuki rika"),
    (1813075, 1471, 1766, "kagaya_yuna", "yuuki rika"),
    (1813075, 1767, 2000, "iyori_ao", "yuuki rika"),
    (1813082, 1, 11, "iyori_ao", "yuuki rika"),
    (1877188, 6, 42, "komachi_nonoka", "yuunagi seshina"), 
    (1877188, 43, 105, "amagamine_renna", "yuuki rika"),
    (1877188, 106, 139, "kagaya_yuna", "yuuki rika"),
    (1921945, 2, 56, "hanamaru_mari", "fuyuichi monme"),
    (1921945, 57, 92, "harizome_shione", "fuyuichi monme"),
    (1921945, 93, 130, "iyori_ao", "yuuki rika"),
    (1921945, 131, 160, None, "fuyuichi monme"),
    (1921951, 8, 62, "hanamaru_mari", "fuyuichi monme"),
    (1921951, 63, 98, "harizome_shione", "fuyuichi monme"),
    (1921951, 99, 136, "iyori_ao", "yuuki rika"),
    (1921951, 137, 166, None, "fuyuichi monme"),

    (2340229, 10, 121, "hanamaru_mari", "fuyuichi monme"),
    (2340229, 122, 246, "harizome_shione", "fuyuichi monme"),
    (2340229, 247, 369, "komachi_nonoka", "yuunagi seshina"),
    (2340229, 370, 496, "amagamine_renna", "yuuki rika"),
    (2340229, 513, 530, "iyori_ao", "yuuki rika"),
    (2340229, 548, 608, "kagaya_yuna", "yuuki rika"),
    # Suki to Suki to de Sankaku Renai
    (998647, 3, 77, "komorie_nanaru", "fuyuichi monme"),
    (998647, 80, 158, "komorie_suzu", "fuyuichi monme"),
    (998647, 159, 264, "narutaki_maho", "yuuki rika"),
    (998647, 265, 357, None, "yuunagi seshina"),

    (1423851, 3, 78, "komorie_nanaru", "fuyuichi monme"),
    (1423851, 81, 159, "komorie_suzu", "fuyuichi monme"),
    (1423851, 160, 265, "narutaki_maho", "yuuki rika"),
    (1423851, 266, 358, None, "yuunagi seshina"),
    # Karigurashi Renai
    (1205109, 3, 125, "sakuragibashi_rito", "yuuki rika"),
    (1205109, 126, 240, "aranami_kyou", "fuyuichi monme"),
    (1205109, 241, 420, "niizuma_hiyori", "fuyuichi monme"),
    (1205109, 421, 580, "yohakari_ayaka", "yuuki rika"),
    (1205109, 581, 658, None, "yuunagi seshina"),
    # Sorairo Innocent
    (882267, 3, 90, "tsukigase_mahiru", "unasaka"),
    (882267, 91, 157, "tsubaki_ami", "unasaka"),
    (1420499, 2, 631, "tsukigase_mahiru", "unasaka"),
    (1420499, 632, 1035, "tsubaki_ami", "unasaka"),
    # Kanojo to Ore no Lovely Day
    (1134208, 1, 192, "mashiro_yuka", "chikotam"),
    (1134208, 193, 384, "kongou_alice", "chikotam"),
    (1134208, 385, 582, "akamine_shizuka", "narumi yuu"),
    (1134208, 583, 780, "aoi_haruka", "narumi yuu"),

    (1023186, 3, 186, "mashiro_yuka", "chikotam"),
    (1023186, 187, 359, "kongou_alice", "chikotam"),
    (1023186, 360, 539, "akamine_shizuka", "narumi yuu"),
    (1023186, 540, 706, "aoi_haruka", "narumi yuu"),
    (1023186, 725, 768, "mashiro_yuka", "chikotam"),
    (1023186, 769, 806, "kongou_alice", "chikotam"),
    (1023186, 807, 835, "akamine_shizuka", "narumi yuu"),
    (1023186, 836, 880, "aoi_haruka", "narumi yuu"),
    # LOVEREC.
    (827841, 32, 60, "yanase_hitomi", "primil"),
    (827841, 61, 95, "shirosawa_miyuki", "narumi yuu"),
    (827841, 96, 122, "mikuriya_nori", "chikotam"),
    (827841, 123, 148, "yoshinaga_chiho", "narumi yuu"),
    (827841, 217, 232, "yanase_hitomi", "primil"),
    (827841, 360, 437, "shirosawa_miyuki", "narumi yuu"),
    (827841, 468, 563, "mikuriya_nori", "chikotam"),
    (827841, 603, 696, "yoshinaga_chiho", "narumi yuu"),
    (899895, 3, 26, "yanase hitomi", "primil"),
    (899895, 27, 47, "shirosawa miyuki", "narumi yuu"),
    (899895, 48, 66, "mikuriya nori", "chikotam"),
    (899895, 67, 90, "yoshinaga chiho", "narumi yuu"),
    # Naka no Hito nado Inai! Tokyo Hero Project
    (522375, 46, 191, "hondou_ayano", "primil"),
    (522375, 298, 440, "amamoto_louis", "primil"),
    (522375, 484, 602, "kirihara_saori", "narumi yuu"),
    (522375, 636, 749, "kamishiro_yuka", "narumi yuu"),

    (1025259, 18, 140, "hondou_ayano", "primil"),
    (1025259, 158, 273, "amamoto_louis", "primil"),
    (1025259, 279, 385, "kirihara_saori", "narumi yuu"),
    (1025259, 393, 498, "kamishiro_yuka", "narumi yuu"),
    # Sakura Iro, Mau Koro ni
    (1389160, 7, 118, "wakiike_koi", "lucie"),
    (1389160, 119, 296, "kuroe", "yuzuka"),
    (1389160, 297, 434, "tomari_mariko", "komeshiro kasu"),
    (1389160, 435, 586, "mizushiro_mina", "yuzuka"),
    (1389160, 591, 744, "kitami_rin", "anapom"),
    (1445230, 1, 400, "mizushiro_mina", "yuzuka"),
    (1445230, 401, 720, "kuroe", "yuzuka"),
    (1445230, 721, 1264, "tomari_mariko", "komeshiro kasu"),
    (1445230, 1265, 1500, "kitami_rin", "anapom"),
    (1445272, 1, 864, "wakiike_koi", "lucie"),
    (1445272, 865, 1080, "hino_yuki", "yuzuka"),
    # Goshujin-sama, Seira ni Yume Mitai na Icha Love Gohoushi Sasete Itadakemasu ka
    (3034052, 1, 2000, "seira_(rubi-sama)", "rubi-sama"),
    (2272848, 1, 2000, "seira_(rubi-sama)", "rubi-sama"),
    # Wan Nyan ☆ A La Mode!
    (887743, 3, 60, "nekohana_korone", "naenae"),
    (887743, 61, 117, "nekodomari_makoto", "rokudou itsuki"),
    (887743, 118, 163, "nekojou_hinana", "wori"),
    (887743, 164, 245, "inukai_shinono", "wori"),
    (887743, 246, 300, "inuta_hana", "naenae"),
    (887743, 301, 359, "inuyama_michiyo", "rubi-sama"),
    (887743, 360, 388, "nekotama_rui", "rubi-sama"),
    (887743, 389, 420, "fumiko_mameshiba_shepherd", "rokudou itsuki"),
    (887743, 443, 466, "nekohana_korone", "naenae"),
    (887743, 467, 487, "nekodomari_makoto", "rokudou itsuki"),
    (887743, 488, 504, "nekojou_hinana", "wori"),
    (887743, 505, 561, "inukai_shinono", "wori"),
    (887743, 562, 608, "inuta_hana", "naenae"),
    (887743, 609, 648, "inuyama_michiyo", "rubi-sama"),
    (887743, 649, 649, "nekotama_rui", "rubi-sama"),
    (887743, 650, 659, "nekohana_korone", "naenae"),
    (887743, 660, 669, "inukai_shinono", "wori"),
    (887743, 670, 686, "inuta_hana", "naenae"),
    (887743, 687, 693, "nekotama_rui", "rubi-sama"),
    (887743, 696, 719, "nekohana_korone", "naenae"),
    (887743, 720, 740, "nekodomari_makoto", "rokudou itsuki"),
    (887743, 741, 757, "nekojou_hinana", "wori"),
    (887743, 758, 814, "inukai_shinono", "wori"),
    (887743, 815, 861, "inuta_hana", "naenae"),
    (887743, 862, 901, "inuyama_michiyo", "rubi-sama"),
    (887743, 902, 902, "nekotama_rui", "rubi-sama"),

    (1131217, 3, 81, "nekohana_korone", "naenae"),
    (1131217, 83, 161, "nekodomari_makoto", "rokudou itsuki"),
    (1131217, 162, 320, "nekojou_hinana", "wori"),
    (1131217, 321, 465, "inukai_shinono", "wori"),
    (1131217, 467, 555, "inuta_hana", "naenae"),
    (1131217, 556, 635, "inuyama_michiyo", "rubi-sama"),
    (1131217, 636, 665, "nekotama_rui", "rubi-sama"),
    (1131217, 666, 685, "fumiko_mameshiba_shepherd", "rokudou itsuki"),

    (1886653, 2, 89, "nekohana_korone", "naenae"),
    (1886653, 90, 171, "nekodomari_makoto", "rokudou itsuki"),
    (1886653, 172, 241, "nekojou_hinana", "wori"),
    (1886653, 242, 406, "inukai_shinono", "wori"),
    (1886653, 407, 525, "inuta_hana", "naenae"),
    (1886653, 526, 627, "inuyama_michiyo", "rubi-sama"),
    (1886653, 628, 662, "nekotama_rui", "rubi-sama"),
    (1886653, 663, 705, "fumiko_mameshiba_shepherd", "rokudou itsuki"),

    (1731737, 7, 68, "nekohana_korone", "naenae"),
    (1731737, 69, 129, "nekodomari_makoto", "rokudou itsuki"),
    (1731737, 130, 182, "nekojou_hinana", "wori"),
    (1731737, 183, 290, "inukai_shinono", "wori"),
    (1731737, 291, 362, "inuta_hana", "naenae"),
    (1731737, 363, 424, "inuyama_michiyo", "rubi-sama"),
    (1731737, 425, 458, "nekotama_rui", "rubi-sama"),
    (1731737, 459, 500, "fumiko_mameshiba_shepherd", "rokudou itsuki"),

    (1735897, 29, 248, "nekohana_korone", "naenae"),
    (1735897, 249, 468, "nekodomari_makoto", "rokudou itsuki"),
    (1735897, 469, 666, "nekojou_hinana", "wori"),
    (1735897, 667, 864, "inukai_shinono", "wori"),
    (1735897, 865, 1044, "inuta_hana", "naenae"),
    (1735897, 1045, 1264, "inuyama_michiyo", "rubi-sama"),
    (1735897, 1265, 1320, "nekotama_rui", "rubi-sama"),
    (1735897, 1321, 1374, "fumiko_mameshiba_shepherd", "rokudou itsuki"),
    # Love Love ♥ Princess
    (839209, 3, 213, "marigold_bruette_erland", "rubi-sama"),
    (839209, 214, 233, "marigold_bruette_erland,  anastasia_imperator_erland", "wori, rubi-sama"),
    (839209, 234, 432, "anastasia_imperator_erland", "wori"),
    (839209, 433, 592, "tsukimori_mio_erland", "rubi-sama"),
    (839209, 593, 736, "fione_riese_erland", "rokudou itsuki"),
    (839209, 737, 885, "cecilia_highland", "rubi-sama"),
    (839209, 886, 991, "paruru_pururu_erland", "wori"),
    (839209, 992, 1070, "angelica_kamira_erland", "rubi-sama"),
    (839209, 1071, 1158, "hinoshita_sakura", "rokudou itsuki"),

    (839731, 3, 407, "marigold_bruette_erland", "rubi-sama"),
    (839731, 408, 719, "anastasia_imperator_erland", "wori"),
    (839731, 720, 989, "tsukimori_mio_erland", "rubi-sama"),
    (839731, 990, 1257, "fione_riese_erland", "rokudou itsuki"),
    (839731, 1258, 1529, "cecilia_highland", "rubi-sama"),
    (839731, 1530, 1879, "paruru_pururu_erland", "wori"),

    (839732, 2, 393, "angelica_kamira_erland", "rubi-sama"),
    (839732, 394, 597, "hinoshita_sakura", "rokudou itsuki"),
    # Love Love Life
    (688579, 2, 124, "akemiya_sakura", "rubi-sama"),
    (688579, 125, 240, "kuroba_kasumi", "wori"),
    (688579, 843, 859, "kuroba_kasumi", "wori"),
    (688579, 241, 376, "leone_goldbach", "wori"),
    (688579, 377, 479, None, "tadima yoshikadu"),
    (688579, 480, 598, "moegino_sachi, moegino_chisa", "wori"),
    (688579, 599, 740, "shizaki_yukari", "rubi-sama"),
    # shona mitsuishi
    (2216911, 2, 10, None, "shona mitsuishi"),
    # Gensou no Idea
    (839435, 2, 90, "nanami_naru", "makita masaki"),
    (839435, 100, 169, "nanami_naru", "makita masaki"),
    (839435, 176, 401, "shinomori_rinon", "makita masaki"),
    (839435, 402, 420, "kenzaki_noel", "makita masaki"),
    (839435, 531, 591, "kenzaki_noel", "makita masaki"),
    (839435, 592, 755, "kujou_mitsuki", "makita masaki"),
    # SORCERY JOKERS
    (1333954, 2, 56, "kousaki_fiona_annabel", "makita masaki"),
    (1333954, 147, 209, "noah_(sorcery_jokers)", "makita masaki"),
    (840561, 193, 277, "kousaki_fiona_annabel", "makita masaki"),
    (840561, 619, 648, "kousaki_fiona_annabel", "makita masaki"),
    (840561, 182, 192, "noah_(sorcery_jokers)", "makita masaki"),
    (840561, 406, 424, "noah_(sorcery_jokers)", "makita masaki"),
    (840561, 435, 443, "noah_(sorcery_jokers)", "makita masaki"),
    (840561, 577, 586, "noah_(sorcery_jokers)", "makita masaki"),
    (840561, 649, 673, "noah_(sorcery_jokers)", "makita masaki"),
    # Onii-chan, Asa made Zutto Gyu tte Shite!
    (1230398, 2, 459, "onami_sora", "k-ko"),
    (1230398, 460, 932, "onami_akane", "k-ko"),
    (1230398, 933, 1346, "onami_kohaku", "k-ko"),
    (1230398, 1347, 1733, "onami_sumi", "k-ko"),
    (1438799, 3, 195, "onami_sora", "k-ko"),
    (1438799, 196, 377, "onami_akane", "k-ko"),
    (1438799, 378, 562, "onami_kohaku", "k-ko"),
    (1438799, 563, 734, "onami_sumi", "k-ko"),

    (3535981, 2, 459, "onami_sora", "k-ko"),
    (3535981, 460, 932, "onami_akane", "k-ko"),
    (3535981, 933, 1346, "onami_kohaku", "k-ko"),
    (3535981, 1347, 1733, "onami_sumi", "k-ko"),
    (3535982, 2, 126, "onami_sora, onami_akane, onami_kohaku, onami_sumi", "k-ko"),
    (3535982, 245, 359, "onami_sora, onami_akane, onami_kohaku, onami_sumi", "k-ko"),
    (3535982, 1, 2000, None, "k-ko"),
    
    # Yakusoku no Natsu, Mahoroba no Yume
    (1230539, 1, 726, "kamiya_rinka", "hisama kumako"),
    (1230539, 727, 1421, "azuma_nagisa", "chikotam"),
    (1230539, 1422, 2000, "ichinose_serina", "naruse hirofumi"),
    (1230540, 1, 381, "ichinose_serina", "naruse hirofumi"),
    (1230540, 382, 843, "kazami_himari", "narumi yuu"),
    # Hare Nochi Kitto Nanohana Biyori
    (1919557, 3, 97, "ayasaki_nanoka", "chikotam"),
    (1919557, 98, 202, "sakakino_konomi", "chikotam"),
    (1919557, 203, 275, "obara_karin", "sakana"),
    (1919557, 276, 358, "sakuragi_amane", "sakura hanpen"),
    (1919557, 396, 420, "ayasaki_nanoka", "chikotam"),
    (1919557, 425, 455, "sakakino_konomi", "chikotam"),
    (1919557, 456, 477, "obara_karin", "sakana"),
    (1919557, 478, 491, "sakuragi_amane", "sakura hanpen"),

    (733798, 2, 96, "ayasaki_nanoka", "chikotam"),
    (733798, 97, 201, "sakakino_konomi", "chikotam"),
    (733798, 202, 274, "obara_karin", "sakana"),
    (733798, 275, 323, "sakuragi_amane", "sakura hanpen"),
    (733798, 324, 348, "ayasaki_nanoka", "chikotam"),
    (733798, 351, 383, "sakakino_konomi", "chikotam"),
    (733798, 384, 405, "obara_karin", "sakana"),
    (733798, 406, 419, "sakuragi_amane", "sakura hanpen"),
    # Koiimo SWEET☆DAYS
    (494646, 2, 243, "yachiho_aoi", "chikotam"),
    (494646, 244, 491, "yachiho_akane", "chikotam"),
    (494646, 492, 705, "kyoukain_yurika", "sakana"),
    (494646, 706, 929, "yamamiya_ena", "chikotam"),
    (494646, 994, 1013, "kyoukain_yurika", "sakana"),

    (675778, 2, 268, "yachiho_aoi", "chikotam"),
    (675778, 269, 539, "yachiho_akane", "chikotam"),
    (675778, 540, 788, "kyoukain_yurika", "sakana"),
    (675778, 789, 1080, "yamamiya_ena", "chikotam"),

    (1134206, 1, 323, "yachiho_aoi", "chikotam"),
    (1134206, 324, 635, "yachiho_akane", "chikotam"),
    (1134206, 636, 899, "kyoukain_yurika", "sakana"),
    (1134206, 900, 1163, "yamamiya_ena", "chikotam"),
    # Yumekoi 
    (600323, 22, 153, "himenomi_miruku", "chikotam"),
    (600323, 154, 285, "nanamori_kurumi", "chikotam"),
    (600323, 286, 418, "natsume_sakuya", "hinata momo"),
    (600323, 419, 604, "hoshizaki_mei", "inagaki miiko"),
    (600323, 605, 648, "himenomi_miruku", "chikotam"),
    (600323, 649, 689, "nanamori_kurumi", "chikotam"),
    (600323, 690, 722, "natsume_sakuya", "hinata momo"),
    (600323, 723, 789, "hoshizaki_mei", "inagaki miiko"),
    # Delivara!
    (1088159, 2, 135, "mikogami_mikoto", "chikotam"),
    (1088159, 136, 286, "yufu_sumika", "chikotam"),
    (1088159, 287, 382, "kujou_aya", "kino"),
    (1088159, 383, 488, "takachiho_kyouko", "konomi"),

    (1088194, 2, 181, "mikogami_mikoto", "chikotam"),
    (1088194, 182, 481, "yufu_sumika", "chikotam"),
    (1088194, 482, 716, "kujou_aya", "kino"),
    (1088194, 717, 832, "takachiho_kyouko", "konomi"),

    (1088189, 2, 86, "mikogami_mikoto", "chikotam"),
    (1088189, 87, 160, "yufu_sumika", "chikotam"),
    (1088189, 161, 233, "kujou_aya", "kino"),
    (1088189, 234, 279, "takachiho_kyouko", "konomi"),
    (1088189, 369, 548, "mikogami_mikoto", "chikotam"),
    (1088189, 548, 848, "yufu_sumika", "chikotam"),
    (1088189, 849, 1083, "kujou_aya", "kino"),
    (1088189, 1084, 1199, "takachiho_kyouko", "konomi"),
    # pieces
    (1445329, 1, 556, "kimihara_yua", "mikagami mamizu"),
    (1445329, 557, 1088, "takanashi_tsumugi_(pieces)", "mikagami mamizu"),
    (1445329, 1089, 1395, "aino_miori", "mikagami mamizu"),
    (1445367, 1, 217, "aino_miori", "mikagami mamizu"),
    (1445367, 218, 855, "mishiro_arisu", "mikagami mamizu"),

    (1390124, 1, 146, "kimihara_yua", "mikagami mamizu"),
    (1390124, 147, 297, "takanashi_tsumugi_(pieces)", "mikagami mamizu"),
    (1390124, 298, 449, "aino_miori", "mikagami mamizu"),
    (1390124, 450, 578, "mishiro_arisu", "mikagami mamizu"),

    (1647868, 6, 143, "kimihara_yua", "mikagami mamizu"),
    (1647868, 144, 261, "takanashi_tsumugi_(pieces)", "mikagami mamizu"),
    (1647868, 262, 373, "aino_miori", "mikagami mamizu"),
    (1647868, 374, 488, "mishiro_arisu", "mikagami mamizu"),
    # Unless Terminalia
    (2175956, 2, 169, "mikuriya_ren", "mikagami mamizu"),
    (2175956, 170, 332, "rina_(unless_terminalia)", "mikagami mamizu"),
    (2175956, 333, 504, "tachibana_charon_(unless_terminalia)", "mikagami mamizu"),
    (2175956, 505, 679, "lucia_valignano", "mikagami mamizu"),
    (2177903, 1, 498, "mikuriya_ren", "mikagami mamizu"),
    (2177903, 499, 1758, "rina_(unless_terminalia)", "mikagami mamizu"),
    (2177903, 1759, 2000, "tachibana_charon_(unless_terminalia)", "mikagami mamizu"),
    (2177905, 1, 822, "tachibana_charon_(unless_terminalia)", "mikagami mamizu"),
    (2177905, 823, 1350, "lucia_valignano", "mikagami mamizu"),
    # shiratama
    (2616641, 41, 55, None, "shiratama"),
    # Suite Life
    (1159390, 3, 350, "akabane_akari", "ayuma sayu"),
    (1159390, 351, 773, "imai_honoka", "naenae"),
    (1159390, 779, 918, "kisaragi_miho", "niki"),
    (1159390, 920, 1308, "mizuno_seina", "ayuma sayu"),
    # Hanagane Kanade * Gram
    (2306447, 1, 2000, "kozakura_yui", "ayuma sayu"),
    (2384842, 3, 240, "kozakura_yui", "ayuma sayu"),
    (2384842, 243, 501, "hananoka_sumire", "ayuma sayu"),
    (3012239, 2, 240, "hoshiizumi_kotona", "ayuma sayu"),

    (3533373, 1, 2000, "kozakura_yui", "ayuma sayu"),
    (3533374, 1, 2000, "hananoka_sumire", "ayuma sayu"),
    (3533375, 1, 374, "hoshiizumi_kotona", "ayuma sayu"),
    # amaane
    (1475817, 1, 318, "kujou_alice", "ayuma sayu"),
    (1475817, 481, 1211, "kujou_alice", "ayuma sayu"),
    # Deep Love Diary
    (990151, 1, 2000, "kitasono_chika", "ayuma sayu"),
    # Abnormal Lovers
    (1149861, 2, 227, "asahina_seri", "ayuma sayu"),
    (1149861, 228, 424, "onodera_akeno", "mayusaki yuu"),
    (1149861, 425, 612, "shimamiya_mimi", "ayuma sayu"),
    (1149861, 613, 767, "kaburagi_yukana", "mayusaki yuu"),
    (1149861, 768, 873, "asahina_seri", "ayuma sayu"),
    (1149861, 874, 966, "onodera_akeno", "mayusaki yuu"),
    (1149861, 967, 1059, "shimamiya_mimi", "ayuma sayu"),
    (1149861, 1060, 1138, "kaburagi_yukana", "mayusaki yuu"),

    (1150058, 1, 552, "asahina_seri", "ayuma sayu"),
    (1150058, 553, 1225, "kaburagi_yukana", "mayusaki yuu"),
    (1150060, 1, 671, "onodera_akeno", "mayusaki yuu"),
    (1150060, 672, 1185, "shimamiya_mimi", "ayuma sayu"),
    (1150060, 1186, 1917, "asahina_seri", "ayuma sayu"),
    # Love of Renai Koutei of LOVE!
    (598819, 199, 558, "ootori_erika", "ozora ituki"),
    (2188762, 200, 559, "ootori_erika", "ozora ituki"),
    # Aikotoba
    (1507163, 1, 2000, "kinoshita_uzuki", None),
    (1536531, 1, 2000, "kinoshita_uzuki", None),
    (1537306, 1, 920, "kinoshita_uzuki", None),
    # Pretty x Cation
    (1190937, 8, 356, "asagiri_nozomi", None),
    (1190937, 357, 715, " asagiri_sakura", None),
    (1190937, 716, 1074, "electrichka_sapsan", None),
    (1190937, 1075, 1422, "yakuouji_komachi", None),

    (1414110, 1, 39, "asagiri_nozomi", None),
    (1414110, 40, 80, "asagiri_sakura", None),
    (1414110, 81, 126, "electrichka_sapsan", None),
    (1414110, 127, 173, "yakuouji_komachi", None),

    (1414456, 1, 754, "asagiri_nozomi", None),
    (1414456, 755, 1650, "asagiri_sakura", None),
    (1414456, 1651, 2000, "electrichka_sapsan", None),
    (1414495, 1, 490, "electrichka_sapsan", None),
    (1414495, 491, 1244, "yakuouji_komachi", None),

    (696253, 2, 206, "asagiri_nozomi", None),
    (696253, 207, 414, "asagiri_sakura", None),
    (696253, 415, 625, "electrichka_sapsan", None),
    (696253, 626, 833, "yakuouji_komachi", None),
    
    (846368, 2, 384, "asagiri_nozomi", None),
    (846368, 385, 777, "asagiri_sakura", None),
    (846368, 778, 1171, "electrichka_sapsan", None),
    (846368, 1172, 1560, "yakuouji_komachi", None),
    # Cocoro＠Function
    (641202, 39, 325, "hasugase_mina", "motoi ayumu"),
    (641202, 893, 1169, "hayami_asagao", "motoi ayumu"),
    (641202, 326, 892, None, "hinata momo"),

    (762285, 2, 21, "hayami_asagao", "motoi ayumu"),
    (762285, 35, 128, "hayami_asagao", "motoi ayumu"),
    (762285, 129, 231, None, "muu"),
    (762285, 232, 469, "hayami_asagao", "motoi ayumu"),
    (762285, 470, 619, None, "muu"),
    (762285, 625, 802, None, "hinata momo"),
    (762285, 803, 970, None, "muu, hinata momo"),
    (762285, 971, 1107, "hasugase_mina", "motoi ayumu"),
    (762285, 1108, 2000, None, "hinata momo"),

    (1076280, 38, 188, "hayami_asagao", "motoi ayumu"),
    (1076280, 189, 349, None, "muu"),
    (1076280, 350, 430, "hayami_asagao", "motoi ayumu"),
    (1076280, 647, 693, "hayami_asagao", "motoi ayumu"),
    (1076280, 694, 940, None, "muu"),
    (1076280, 948, 1300, None, "hinata momo"),
    (1076280, 1301, 1431, None, "muu, hinata momo"),
    (1076280, 1432, 1673, "hasugase_mina", "motoi ayumu"),
    (1076280, 1674, 2000, None, "hinata momo"),

    # motoi ayumu
    (491097, 273, 876, None, "motoi ayumu"),
    (491097, 1221, 1249, None, "motoi ayumu"),
    (491097, 1317, 1347, None, "motoi ayumu"),
    (491097, 1, 2000, None, "yashima takahiro"),
    (634769, 47, 58, None, "motoi ayumu"),
    (634769, 132, 276, None, "motoi ayumu"),
    (634769, 415, 470, None, "motoi ayumu"),
    (634769, 634, 881, None, "motoi ayumu"),
    (634769, 1, 2000, None, "yashima takahiro"),
    (1907517, 3, 164, None, "motoi ayumu"),
    (1907517, 328, 545, None, "motoi ayumu"),
    (1907517, 1, 2000, None, "yashima takahiro"),
    # Koikishi Purely ☆ Kiss
    (875317, 3, 271, "kazama_akari", "yuuki hagure"),
    (875317, 320, 708, "shidou_mana", "yuuki hagure"),
    (875317, 725, 805, "bernadette_villeburg", "yuuki hagure"),
    (875317, 806, 880, "kuninaka_kaori", "yuuki hagure"),
    (875317, 881, 1273, "elcia_harvence", "yuuki hagure"),
    (875317, 1312, 1667, "fujimori_yuu", "yuuki hagure"),

    (1302133, 1, 558, "elcia_harvence", "yuuki hagure"),
    (1302133, 559, 936, "bernadette_villeburg", "yuuki hagure"),
    (1302133, 937, 1368, "kuninaka_kaori", "yuuki hagure"),
    (1302133, 1417, 1936, "shidou_mana", "yuuki hagure"),
    (1302133, 1937, 2000, "kazama_akari", "yuuki hagure"),
    (1302134, 1, 429, "kazama_akari", "yuuki hagure"),
    (1302134, 430, 913, "fujimori_yuu", "yuuki hagure"),
    # Juukishi Cutie ☆ Bullet
    (840881, 2, 85, "minami_mayu", "yuuki hagure"),
    (840881, 86, 142, "fujikura_miyabi", "yuuki hagure"),
    (840881, 143, 162, "tanegashima_wakasa", "yuuki hagure"),
    (840881, 163, 266, "reina_de_medishi", "yuuki hagure"),
    (840881, 267, 332, "sara_tefal", "yuuki hagure"),

    (1868156, 8, 547, "sara_tefal", "yuuki hagure"),
    (1868156, 756, 1100, "reina_de_medishi", "yuuki hagure"),
    (1868156, 1101, 1330, "tanegashima_wakasa", "yuuki hagure"),
    (1868156, 1331, 1707, "fujikura_miyabi", "yuuki hagure"),
    (1868156, 1836, 2000, "minami_mayu", "yuuki hagure"),
    (1868168, 1, 220, "minami_mayu", "yuuki hagure"),
    (1868168, 221, 412, "yuuna_de_medishi", "yuuki hagure"),
    # kimagure temptation 
    (1477239, 1, 26, "anneliese", "kimishima ao"),
    (1491062, 1, 212, "anneliese", "kimishima ao"),
    (2065814, 1, 216, "anneliese", "kimishima ao"),
    (3139962, 56, 133, "anneliese", "kimishima ao"),
    (3139962, 136, 137, "anneliese", "kimishima ao"),
    (3139962, 156, 279, "anneliese", "kimishima ao"),
    # Koisuru Kimochi no Kasanekata
    (877931, 2, 104, "tsukishima_saori", "kimishima ao"),
    (877931, 105, 211, "hiiragi_mio", "kimishima ao"),
    (877931, 212, 336, "narumi_akane", "kimishima ao"),
    (877931, 337, 441, "kuonji_hiyori", "kimishima ao"),
    (877931, 442, 627, "kaburagi_yukie", "kimishima ao"),
    (877931, 628, 656, "ougi_ichika", "kimishima ao"),

    (948403, 3, 86, "tsukishima_saori", "kimishima ao"),
    (948403, 87, 151, "hiiragi_mio", "kimishima ao"),
    (948403, 152, 231, "narumi_akane", "kimishima ao"),
    (948403, 232, 303, "kuonji_hiyori", "kimishima ao"),
    (948403, 304, 394, "kaburagi_yukie", "kimishima ao"),
    (948403, 395, 472, "ougi_ichika", "kimishima ao"),

    (902543, 1, 78, "hiiragi_mio", "kimishima ao"),
    (902543, 79, 160, "kuonji_hiyori", "kimishima ao"),

    (2843300, 1, 65, "kaburagi_yukie", "kimishima ao"),
    (2843300, 66, 118, "ougi_ichika", "kimishima ao"),
    # Otome ga Kanaderu Koi no Aria
    (1003525, 843, 903, "jougasaki_kanade", "kimishima ao"),
    (827706, 116, 157, "jougasaki_kanade", "kimishima ao"),
    (827706, 623, 657, "jougasaki_kanade", "kimishima ao"),
    (827706, 765, 799, "jougasaki_kanade", "kimishima ao"),
    (827706, 907, 972, "jougasaki_kanade", "kimishima ao"),

    # Haze Man
    (1121739, 17, 94, "fara_perelreese", "kyou"),
    (1121739, 110, 167, "miyadera_renka", "kyou"),
    (1121739, 171, 173, "miyadera_renka", "kyou"),
    (1121739, 209, 241, "fara_perelreese, miyadera_renka", "kyou"),

    # D.S. -Dal Segno
    (1083084, 2, 135, "asamiya_himari", "tanihara natsuki"),
    (1083084, 143, 278, "murasaki_hazuki" ,"tanihara natsuki"),
    (1083084, 279, 414, "kouzuki_io" ,"takano yuki"),
    (1083084, 415, 577, "fujishiro_noeri" ,"takano yuki"),
    (1083084, 578, 685, "ame_(d.s. -dal segno-)", "takano yuki"),

    (1056040, 4, 49, "ame_(d.s. -dal segno-)", "takano yuki"),
    (1056040, 65, 177, "murasaki_hazuki", "tanihara natsuki"),
    (1056040, 178, 281, "asamiya_himari", "tanihara natsuki"),
    (1056040, 282, 378, "kouzuki_io", "takano yuki"),
    (1056040, 379, 482, "fujishiro_noeri", "takano yuki"),
    (1056040, 483, 486,"ame_(d.s. -dal segno-)", "takano yuki"),
    (1056040, 487, 499, "murasaki_hazuki", "tanihara natsuki"),
    (1056040, 500, 507, "asamiya_himari", "tanihara natsuki"),
    (1056040, 508, 511, "kouzuki_io", "takano yuki"),
    (1056040, 512, 515, "fujishiro_noeri", "takano yuki"),

    (929047, 4, 111, "ame_(d.s. -dal segno-)", "takano yuki"),
    (929047, 138, 273, "murasaki_hazuki" ,"tanihara natsuki"),
    (929047, 284, 414, "asamiya_himari", "tanihara natsuki"),
    (929047, 415, 550, "kouzuki_io" ,"takano yuki"),
    (929047, 551, 717, "fujishiro_noeri" ,"takano yuki"),
    # D.C.4 ~Da Capo 4~
    (1994876, 1, 126, "sagisawa_arisu", "tanihara natsuki"),
    (1994876, 142, 200, "shirakawa_hiyori", "takano yuki"),
    (1994876, 201, 247, "mishima_miu", "kisaragi yuu"),
    (1994876, 248, 297, "tokisaka_nino", "takano yuki"),
    (1994876, 298, 366, "houjou_shiina", "mitsumomo mam"),
    (1994876, 367, 423, "oumi_sorane", "tanihara natsuki"),
    (1994876, 424, 466, "hinohara_chiyoko", "tanihara natsuki"),
    (1994876, 467, 585, "sagisawa_arisu", "tanihara natsuki"),
    (1994876, 586, 651, "shirakawa_hiyori", "takano yuki"),
    (1994876, 652, 696, "mishima_miu", "kisaragi yuu"),
    (1994876, 697, 769, "tokisaka_nino", "takano yuki"),
    (1994876, 770, 829, "houjou_shiina", "mitsumomo mam"),
    (1994876, 830, 904, "oumi_sorane", "tanihara natsuki"),
    (1994876, 905, 964, "hinohara_chiyoko", "tanihara natsuki"),

    (1994876, 989, 990, "sagisawa_arisu", "tanihara natsuki"),
    (1994876, 991, 991, "tokisaka_nino", "takano yuki"),
    (1994876, 992, 992, "oumi_sorane", "tanihara natsuki"),
    (1994876, 993, 993, "shirakawa_hiyori", "takano yuki"),
    (1994876, 994, 994, "houjou_shiina", "mitsumomo mam"),
    (1994876, 995, 995, "mishima_miu", "kisaragi yuu"),
    (1994876, 996, 996, "hinohara_chiyoko", "tanihara natsuki"),

    (2205861, 1, 73, "sagisawa_arisu", "tanihara natsuki"),
    (2205861, 98, 125, "shirakawa_hiyori", "takano yuki"),
    (2205861, 126, 153, "mishima_miu", "kisaragi yuu"),
    (2205861, 154, 184, "tokisaka_nino", "takano yuki"),
    (2205861, 185, 221, "houjou_shiina", "mitsumomo mam"),
    (2205861, 222, 249, "oumi_sorane", "tanihara natsuki"),
    (2205861, 250, 279, "hinohara_chiyoko", "tanihara natsuki"),
    (2205861, 280, 365, "sagisawa_arisu", "tanihara natsuki"),
    (2205861, 366, 418, "shirakawa_hiyori", "takano yuki"),
    (2205861, 419, 465, "mishima_miu", "kisaragi yuu"),
    (2205861, 466, 516, "tokisaka_nino", "takano yuki"),
    (2205861, 517, 560, "houjou_shiina", "mitsumomo mam"),
    (2205861, 561, 605, "oumi_sorane", "tanihara natsuki"),
    (2205861, 606, 664, "hinohara_chiyoko", "tanihara natsuki"),
    # D.C.5
    (3291043, 3, 57, "shirakawa_aika", "tanihara natsuki"),
    (3291043, 58, 101, "shirakawa_akari", "yatanukikey"),
    (3291043, 113, 160, "yasaka_kako", "takano yuki"),
    (3291043, 161, 203, "yasaka_menoa", "takano yuki"),
    (3291043, 204, 285, "sakuragi_mizuha", "tanihara natsuki"),
    (3291043, 289, 330, "tokisaka_setsuna", "kisaragi yuu"),
    (3291043, 331, 431, "shirakawa_aika", "tanihara natsuki"),
    (3291043, 432, 531, "shirakawa_akari", "yatanukikey"),
    (3291043, 532, 626, "yasaka_kako", "takano yuki"),
    (3291043, 627, 719, "yasaka_menoa", "takano yuki"),
    (3291043, 720, 817, "sakuragi_mizuha", "tanihara natsuki"),
    (3291043, 818, 912, "tokisaka_setsuna", "kisaragi yuu"),

    (3071318, 4, 66, "shirakawa_aika", "tanihara natsuki"),
    (3071318, 92, 164, "yasaka_kako", "takano yuki"),
    (3071318, 165, 222, "yasaka_menoa", "takano yuki"),
    (3071318, 229, 290, "sakuragi_mizuha", "tanihara natsuki"),
    (3071318, 320, 404, "tokisaka_setsuna", "kisaragi yuu"),
    (3071318, 405, 484, "shirakawa_aika", "tanihara natsuki"),
    (3071318, 485, 567, "yasaka_kako", "takano yuki"),
    (3071318, 568, 665, "yasaka_menoa", "takano yuki"),
    (3071318, 666, 762, "sakuragi_mizuha", "tanihara natsuki"),
    (3071318, 763, 848, "tokisaka_setsuna", "kisaragi yuu"),

    (3291121, 7, 61, "shirakawa_aika", "tanihara natsuki"),
    (3291121, 62, 105, "shirakawa_akari", "yatanukikey"),
    (3291121, 117, 164, "yasaka_kako", "takano yuki"),
    (3291121, 165, 207, "yasaka_menoa", "takano yuki"),
    (3291121, 208, 289, "sakuragi_mizuha", "tanihara natsuki"),
    (3291121, 290, 331, "tokisaka_setsuna", "kisaragi yuu"),
    (3291121, 332, 432, "shirakawa_aika", "tanihara natsuki"),
    (3291121, 433, 532, "shirakawa_akari", "yatanukikey"),
    (3291121, 533, 627, "yasaka_kako", "takano yuki"),
    (3291121, 628, 720, "yasaka_menoa", "takano yuki"),
    (3291121, 721, 818, "sakuragi_mizuha", "tanihara natsuki"),
    (3291121, 819, 913, "tokisaka_setsuna", "kisaragi yuu"),
    # Trinoline
    (3418172, 1, 61, "tsumugi_sara", "yuzuna hiyo"),
    (3418172, 68, 127, "tsumugi_sara", "yuzuna hiyo"),
    (3418172, 619, 744, "tsumugi_sara", "yuzuna hiyo"),
    (3418172, 752, 1916, "tsumugi_sara", "yuzuna hiyo"),
    (3418172, 1917, 1970, "nanami_shirone", "konomi"),
    (3418173, 1, 376, "nanami_shirone", "konomi"),
    (3418173, 443, 1211, "nanami_shirone", "konomi"),
    (3418173, 1212, 2000, "miyakaze_yuuri", "konomi"),
    (3418174, 1, 1682, "miyakaze_yuuri", "konomi"),
    (1119329, 296, 668, "tsumugi_sara", "yuzuna hiyo"),
    (1119329, 669, 902, "nanami_shirone", "konomi"),
    (1119329, 903, 2000, "miyakaze_yuuri", "konomi"),

    (3418170, 75, 102, "tsumugi_sara", "yuzuna hiyo"),
    (3418170, 1620, 1778, "tsumugi_sara", "yuzuna hiyo"),
    (3418170, 1780, 2000, "tsumugi_sara", "yuzuna hiyo"),
    (3418170, 114, 204, "nanami_shirone", "konomi"),
    (3418170, 231, 420, "nanami_shirone, tsumugi_sara, miyakaze_yuuri", "konomi, yuzuna hiyo"),
    (3418170, 425, 480, "nanami_shirone", "konomi"),
    (3418170, 550, 686, "nanami_shirone", "konomi"),
    (3418170, 854, 1324, "nanami_shirone", "konomi"),
    (3418170, 14, 51, "miyakaze_yuuri", "konomi"),
    # Trinoline: Genesis
    (1178402, 95, 833, "tsumugi_sara", "yuzuna hiyo"),
    (1178402, 834, 1356, "nanami_shirone", "konomi"),
    (1178402, 1357, 1966, "miyakaze_yuuri", "konomi"),
    (1178441, 1, 66, "miyakaze_yuuri", "konomi"),
    (1178442, 1, 558, "himeno_towa", "yuzuna hiyo"),
    # Sono Hi no Kemono ni wa
    (1354206, 1124, 1133, "tomose_runa", "kaniya shiku"),
    (1354206, 518, 794, "mihama_inori", "yuzuna hiyo"),
    (1354206, 1291, 1608, "mihama_inori", "yuzuna hiyo"),
    (1354206, 1041, 1290, "ikegai_mayu", "konomi"),
    (1354206, 1941, 2000, "ikegai_mayu", "konomi"),
    (1354206, 803, 1040, "tomose_runa", "kaniya shiku"),
    (1354206, 1609, 1940, "tomose_runa", "kaniya shiku"),
    (1354273, 106, 387, "ikegai_mayu", "konomi"),

    (1805418, 1125, 1134, "tomose_runa", "kaniya shiku"),
    (1805418, 519, 795, "mihama_inori", "yuzuna hiyo"),
    (1805418, 1292, 1609, "mihama_inori", "yuzuna hiyo"),
    (1805418, 1042, 1291, "ikegai_mayu", "konomi"),
    (1805418, 1942, 2000, "ikegai_mayu", "konomi"),
    (1805418, 804, 1041, "tomose_runa", "kaniya shiku"),
    (1805418, 1610, 1941, "tomose_runa", "kaniya shiku"),
    (1805418, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (1805420, 107, 388, "ikegai_mayu", "konomi"),
    # 12 no Tsuki no Eve
    (671506, 593, 1039, "unahara_yuki", "yuzuna hiyo"),
    (671506, 1040, 1140, "shiina_mizuka", "takasaki maco"),
    (671507, 1, 305, "shiina_mizuka", "takasaki maco"),
    (671507, 306, 530, "shiina_anzu", "shona mitsuishi"),
    (671507, 531, 677, "shiina_mizuka", "takasaki maco"),
    (671507, 678, 853, "shiina_anzu", "shona mitsuishi"),
    # Yome Sagashi ga Hakadorisugite Yabai.
    (2971469, 2, 134, "yagami_kanna", "ikegami akane"),
    (2971469, 135, 276, "yagami_serika", "ikegami akane"),
    (2971469, 277, 417, "yagami_mihono", "ikegami akane"),
    (2971469, 418, 553, "takamiya_nanaka", "ikegami akane"),
    (2971469, 554, 705, "ashihara_kirino", "ikegami akane"),
    (2971469, 706, 741, "shindou_meika", "ikegami akane"),
    (2971469, 742, 768, "tamano_yui", "ikegami akane"),

    (878236, 3, 135, "yagami_kanna", "ikegami akane"),
    (878236, 136, 278, "yagami_serika", "ikegami akane"),
    (878236, 409, 420, "takamiya_nanaka", "ikegami akane"),
    (878236, 279, 431, "yagami_mihono", "ikegami akane"),
    (878236, 432, 555, "takamiya_nanaka", "ikegami akane"),
    (878236, 556, 706, "ashihara_kirino", "ikegami akane"),
    (878236, 707, 738, "shindou_meika", "ikegami akane"),
    (878236, 739, 784, "tamano_yui", "ikegami akane"),

    (1196589, 1, 105, "yagami_kanna", "ikegami akane"),
    (1196589, 106, 225, "yagami_serika", "ikegami akane"),
    (1196589, 226, 379, "yagami_mihono", "ikegami akane"),
    (1196589, 380, 474, "takamiya_nanaka", "ikegami akane"),
    (1196589, 475, 600, "ashihara_kirino", "ikegami akane"),
    (1196589, 601, 664, "shindou_meika", "ikegami akane"),
    (1196589, 665, 736, "tamano_yui", "ikegami akane"),
    # Deatte 5-fun wa Ore no Mono! Jikan Teishi to Atropos
    (1305605, 7, 170, "mitsui_ruri", "ikegami akane"),
    (1305605, 171, 312, "kurose_sakura", "ikegami akane"),
    (1305605, 313, 456, "yamabuki_noa", "ikegami akane"),
    (1305605, 457, 610, "kurebayashi_kanon", "ikegami akane"),
    (1305605, 611, 772, "hiiragi_hakua", "ikegami akane"),
    (1305605, 773, 830, "shinonome_azusa", "ikegami akane"),
    (1305605, 831, 870, "abe_kurumi", "ikegami akane"),
    (1375991, 1, 144, "mitsui_ruri", "ikegami akane"),
    (1375991, 145, 270, "kurose_sakura", "ikegami akane"),
    (1375991, 271, 396, "yamabuki_noa", "ikegami akane"),
    (1375991, 397, 504, "kurebayashi_kanon", "ikegami akane"),
    (1375991, 505, 666, "hiiragi_hakua", "ikegami akane"),
    (1375991, 667, 726, "shinonome_azusa", "ikegami akane"),
    (1375991, 727, 774, "abe_kurumi", "ikegami akane"),
    # Ore no Hitomi de Maruhadaka
    (2622635, 4, 124, "lucie_stella_ecarlate", "ikegami akane"),
    (2622635, 125, 244, "yaezakura_koume", "ikegami akane"),
    (2622635, 245, 359, "kurusu_hikaru", "ikegami akane"),
    (2622635, 360, 498, "itsukage_honoka", "ikegami akane"),
    (2622635, 499, 612, "eliska_fortinova", "ikegami akane"),
    (2622635, 613, 643, "noumi_mizuha", "ikegami akane"),
    (2622635, 644, 675, "tatsumiya_homura", "ikegami akane"),

    (3590370, 1, 126, "lucie_stella_ecarlate", "ikegami akane"),
    (3590370, 127, 353, "yaezakura_koume", "ikegami akane"),
    (3590370, 354, 486, "kurusu_hikaru", "ikegami akane"),
    (3590370, 487, 736, "itsukage_honoka", "ikegami akane"),
    (3590370, 737, 862, "eliska_fortinova", "ikegami akane"),
    (3590370, 863, 950, "noumi_mizuha", "ikegami akane"),
    (3590370, 951, 1078, "tatsumiya_homura", "ikegami akane"),
    # Ore no Sugata ga, Toumei ni
    (1596946, 6, 162, "matsuyuki_ame", "ikegami akane"),
    (1596946, 163, 314, "kuzunoha_chitose", "ikegami akane"),
    (1596946, 315, 454, "yotsuba_kohaku", "ikegami akane"),
    (1596946, 455, 593, "ayame_nayuri", "ikegami akane"),
    (1596946, 594, 754, "takashiro_towa", "ikegami akane"),
    (1596946, 755, 791, "matsuyuki_kiri", "ikegami akane"),
    (1596946, 792, 804, "minazuki_shio", "ikegami akane"),

    (1597545, 274, 493, "matsuyuki_ame", "ikegami akane"),
    (1597545, 1, 273, "kuzunoha_chitose", "ikegami akane"),
    (1597545, 894, 1093, "yotsuba_kohaku", "ikegami akane"),
    (1597545, 494, 693, "ayame_nayuri", "ikegami akane"),
    (1597545, 694, 893, "takashiro_towa", "ikegami akane"),
    (1597545, 1094, 1191, "matsuyuki_kiri", "ikegami akane"),
    (1597545, 1192, 1331, "minazuki_shio", "ikegami akane"),
    # KamiYaba: Destiny on a Dicey Deadline/Kami tanomishi sugi de ore no mirai ga yabai
    (2969351, 4, 154, "nagumo_nanami", "ikegami akane"),
    (2969351, 155, 314, "akagi_suzuna", "ikegami akane"),
    (2969351, 315, 465, "suou_yukari", "ikegami akane"),
    (2969351, 466, 617, "urara_(kamiyaba)", "ikegami akane"),
    (2969351, 618, 686, "sakashiro_hanayo", "ikegami akane"),
    (2969351, 687, 760, "kanbayashi_mao", "ikegami akane"),

    (1043628, 4, 245, "nagumo_nanami", "ikegami akane"),
    (1043628, 246, 405, "akagi_suzuna", "ikegami akane"),
    (1043628, 406, 556, "suou_yukari", "ikegami akane"),
    (1043628, 557, 708, "urara_(kamiyaba)", "ikegami akane"),
    (1043628, 709, 777, "sakashiro_hanayo", "ikegami akane"),
    (1043628, 778, 850, "kanbayashi_mao", "ikegami akane"),

    (3590407, 1, 126, "nagumo_nanami", "ikegami akane"),
    (3590407, 127, 240, "akagi_suzuna", "ikegami akane"),
    (3590407, 241, 384, "suou_yukari", "ikegami akane"),
    (3590407, 385, 546, "urara_(kamiyaba)", "ikegami akane"),
    (3590407, 547, 650, "sakashiro_hanayo", "ikegami akane"),
    (3590407, 651, 715, "kanbayashi_mao", "ikegami akane"),

    # Ore no Cupid ga Ponkotsu Sugite Kowa~i
    (2028649, 2, 168, "koimi_eiru", "ikegami akane"),
    (2028649, 169, 337, "inui_isuzu", "ikegami akane"),
    (2028649, 338, 426, "minekoshi_kotohina", "ikegami akane"),
    (2028649, 427, 509, "kousaki_kururu", "ikegami akane"),
    (2028649, 513, 671, "saijou_rinka", "ikegami akane"),
    (2028649, 672, 870, "sakuma_uta", "ikegami akane"),

    (2619768, 2, 45, "koimi_eiru", "ikegami akane"),
    (2619768, 46, 85, "inui_isuzu", "ikegami akane"),
    (2619768, 86, 112, "kousaki_kururu", "ikegami akane"),
    (2619768, 115, 150, "saijou_rinka", "ikegami akane"),
    (2619768, 151, 208, "sakuma_uta", "ikegami akane"),
    (2619768, 209, 236, "minekoshi_kotohina", "ikegami akane"),
    (2619768, 240, 366, "koimi_eiru", "ikegami akane"),
    (2619768, 367, 502, "inui_isuzu", "ikegami akane"),
    (2619768, 503, 560, "kousaki_kururu", "ikegami akane"),
    (2619768, 561, 683, "saijou_rinka", "ikegami akane"),
    (2619768, 684, 826, "sakuma_uta", "ikegami akane"),
    (2619768, 827, 888, "minekoshi_kotohina", "ikegami akane"),
    (2619768, 889, 932, "koimi_eiru", "ikegami akane"),
    (2619768, 933, 972, "inui_isuzu", "ikegami akane"),
    (2619768, 973, 999, "kousaki_kururu", "ikegami akane"),
    (2619768, 1002, 1037, "saijou_rinka", "ikegami akane"),
    (2619768, 1038, 1095, "sakuma_uta", "ikegami akane"),
    (2619768, 1096, 1123, "minekoshi_kotohina", "ikegami akane"),
    (2619768, 1127, 1253, "koimi_eiru", "ikegami akane"),
    (2619768, 1254, 1389, "inui_isuzu", "ikegami akane"),
    (2619768, 1390, 1447, "kousaki_kururu", "ikegami akane"),
    (2619768, 1448, 1570, "saijou_rinka", "ikegami akane"),
    (2619768, 1571, 1713, "sakuma_uta", "ikegami akane"),
    (2619768, 1714, 1775, "minekoshi_kotohina", "ikegami akane"),


    (2620666, 2, 46, "koimi_eiru", "ikegami akane"),
    (2620666, 47, 86, "inui_isuzu", "ikegami akane"),
    (2620666, 87, 113, "kousaki_kururu", "ikegami akane"),
    (2620666, 116, 151, "saijou_rinka", "ikegami akane"),
    (2620666, 152, 209, "sakuma_uta", "ikegami akane"),
    (2620666, 210, 237, "minekoshi_kotohina", "ikegami akane"),
    (2620666, 241, 367, "koimi_eiru", "ikegami akane"),
    (2620666, 368, 503, "inui_isuzu", "ikegami akane"),
    (2620666, 504, 561, "kousaki_kururu", "ikegami akane"),
    (2620666, 562, 684, "saijou_rinka", "ikegami akane"),
    (2620666, 685, 826, "sakuma_uta", "ikegami akane"),
    (2620666, 827, 888, "minekoshi_kotohina", "ikegami akane"),

    (3590447, 1, 231, "koimi_eiru", "ikegami akane"),
    (3590447, 232, 331, "inui_isuzu", "ikegami akane"),
    (3590447, 332, 436, "kousaki_kururu", "ikegami akane"),
    (3590447, 437, 541, "saijou_rinka", "ikegami akane"),
    (3590447, 542, 673, "sakuma_uta", "ikegami akane"),
    (3590447, 674, 725, "minekoshi_kotohina", "ikegami akane"),
    # Kokoro no Katachi to Iro to Oto
    (1808643, 1, 211, "hoshina_harune", "ikegami akane"),
    (3590404, 1, 153, "hoshina_harune", "ikegami akane"),
    # Imouto no Okage de Motesugite Yabai.
    (742094, 1, 123, "shiratori_kanae", "ikegami akane"),
    (742094, 124, 225, "toshima_maina", "ikegami akane"),
    (742094, 226, 328, "yonaga_aoba", "ikegami akane"),
    (742094, 329, 432, "mizunashi_miya", "ikegami akane"),
    (742094, 433, 544, "asasaka_meguri", "ikegami akane"),
    (742094, 545, 569, "aisaka_sarina", "ikegami akane"),
    (742094, 570, 593, "himemiya_yurika", "ikegami akane"),

    (1960844, 1, 138, "shiratori_kanae", "ikegami akane"),
    (1960844, 139, 299, "toshima_maina", "ikegami akane"),
    (1960844, 300, 467, "yonaga_aoba", "ikegami akane"),
    (1960844, 468, 621, "mizunashi_miya", "ikegami akane"),
    (1960844, 622, 821, "asasaka_meguri", "ikegami akane"),
    (1960844, 822, 845, "aisaka_sarina", "ikegami akane"),
    (1960844, 846, 869, "himemiya_yurika", "ikegami akane"),

    (733772, 2, 57, "shiratori_kanae", "ikegami akane"),
    (733772, 58, 120, "asasaka_meguri", "ikegami akane"),

    (2655087, 1, 56, "shiratori_kanae", "ikegami akane"),
    (2655087, 57, 119, "asasaka_meguri", "ikegami akane"),

    (3590371, 1, 46, "shiratori_kanae", "ikegami akane"),
    (3590371, 47, 171, "asasaka_meguri", "ikegami akane"),
    # Natsuiro Recipe
    (819570, 157, 279, "yaehara_yuzu", "non"),
    # Hatsukoi Sankaime
    (1009352, 2, 69, "kusunoha_misaki", "chikotam"),
    (1009352, 74, 88, None, "chikotam"),
    (1009352, 89, 363, "kusunoha_misaki", "chikotam"),
    (1009352, 2, 363, None, "chikotam"),
    (1009352, 364, 813, "hikami_yurino", "narumi yuu"),
    (1009352, 814, 1238, "emiliya_karimov", "narumi yuu"),
    (1009352, 1239, 1596, "yanagihara_sui", "takashina at masato"),
    (1009352, 1598, 1919, "kushiro_himeka", "takashina at masato"),
    # Hakoniwa Logic
    (753681, 4, 103, "maezono_kirika", "yukie"),
    (753681, 106, 217, "iriya_koko", "yukie"),
    (753681, 218, 322, "kidou_shizuku", "miwa futaba"),
    (753681, 323, 449, "amesara_mana", "miwa futaba"),
    (753681, 450, 537, "sakuraba_moemi", "yukie"),

    (948531, 2, 164, "iriya_koko", "yukie"),

    (3595341, 1, 810, "maezono_kirika", "yukie"),
    (3595341, 811, 1440, "iriya_koko", "yukie"),
    (3595341, 1441, 1690, "kidou_shizuku", "miwa futaba"),
    (3595466, 511, 850, "amesara_mana", "miwa futaba"),
    (3595466, 1, 510, "sakuraba_moemi", "yukie"),
    # QUINTUPLE☆SPLASH
    (886805, 2, 79, "yashio_sahori", "sakana"),
    (886805, 80, 216, "sakuraura_sae", "mikeou"),
    (886805, 217, 343, "moriya_ami", "yukie"),
    (886805, 344, 446, "misato_mio", "ichiri"),
    (886805, 447, 538, "minori_tomoe", "sakura hanpen"),

    (886805, 539, 563, "yashio_sahori", "sakana"),
    (886805, 564, 587, "sakuraura_sae", "mikeou"),
    (886805, 588, 615, "moriya_ami", "yukie"),
    (886805, 616, 640, "misato_mio", "ichiri"),
    (886805, 641, 737, "minori_tomoe", "sakura hanpen"),

    (1134306, 1, 224, "yashio_sahori", "sakana"),
    (1134306, 225, 672, "sakuraura_sae", "mikeou"),
    (1134306, 673, 1120, "moriya_ami", "yukie"),
    (1134306, 1121, 1582, "misato_mio", "ichiri"),
    (1134306, 1583, 1822, "minori_tomoe", "sakura hanpen"),

    (947683, 1, 17, "yashio_sahori", "sakana"),
    (947683, 22, 39, "misato_mio", "ichiri"),

    (2313627, 1, 2000, None, "mizuno sao, satasama, yuzuna hiyo"),
    (1537715, 1, 2000, None, "yuzuna hiyo, mizuno sao, satasama"),
    (1537491, 1, 2000, None, "yuzuna hiyo, mizuno sao, satasama"),
    (1537567, 1, 2000, None, "yuzuna hiyo, mizuno sao, satasama"),
    (1537457, 1, 2000, None, "yuzuna hiyo, mizuno sao, satasama"),
    (793088, 1, 2000, None, "yuzuna hiyo, mizuno sao, satasama"),
    # Sakura Hitohira Koi Moyou
    (1122203, 4, 172, "kamikawa_saya", "sakura hanpen"),
    (1122203, 173, 407, "mizutani_yoshino", "ichiri"),
    (1122203, 408, 605, "takazawa_miaya", "sakura hanpen"),
    (1122203, 606, 801, "hatsushiba_chitose", "ichiri"),
    (1122203, 802, 867, "kamikawa_saya", "sakura hanpen"),
    (1122203, 868, 944, "mizutani_yoshino", "ichiri"),
    (1122203, 945, 1011, "takazawa_miaya", "sakura hanpen"),
    (1122203, 1012, 1083, "hatsushiba_chitose", "ichiri"),

    (1134207, 1, 222, "kamikawa_saya", "sakura hanpen"),
    (1134207, 223, 414, "mizutani_yoshino", "ichiri"),
    (1134207, 415, 606, "takazawa_miaya", "sakura hanpen"),
    (1134207, 607, 798, "hatsushiba_chitose", "ichiri"),

    (1469290, 12, 77, "kamikawa_saya", "sakura hanpen"),
    (1469290, 78, 154, "mizutani_yoshino", "ichiri"),
    (1469290, 155, 221, "takazawa_miaya", "sakura hanpen"),
    (1469290, 222, 293, "hatsushiba_chitose", "ichiri"),
    (1469290, 303, 471, "kamikawa_saya", "sakura hanpen"),
    (1469290, 472, 706, "mizutani_yoshino", "ichiri"),
    (1469290, 707, 904, "takazawa_miaya", "sakura hanpen"),
    (1469290, 905, 1100, "hatsushiba_chitose", "ichiri"),

    (1469322, 69, 253, "kamikawa_saya", "sakura hanpen"),
    (1469322, 254, 413, "mizutani_yoshino", "ichiri"),
    (1469322, 414, 573, "takazawa_miaya", "sakura hanpen"),
    (1469322, 574, 733, "hatsushiba_chitose", "ichiri"),
    (1469322, 856, 1040, "kamikawa_saya", "sakura hanpen"),
    (1469322, 1643, 2000, "kamikawa_saya", "sakura hanpen"),
    (1469322, 1041, 1200, "mizutani_yoshino", "ichiri"),
    (1469322, 1201, 1360, "takazawa_miaya", "sakura hanpen"),
    (1469322, 1361, 1520, "hatsushiba_chitose", "ichiri"),

    (1469320, 430, 466, "kamikawa_saya", "sakura hanpen"),
    (1469320, 467, 498, "mizutani_yoshino", "ichiri"),
    (1469320, 499, 530, "takazawa_miaya", "sakura hanpen"),
    (1469320, 531, 562, "hatsushiba_chitose", "ichiri"),
    (1469320, 591, 627, "kamikawa_saya", "sakura hanpen"),
    (1469320, 628, 659, "mizutani_yoshino", "ichiri"),
    (1469320, 660, 691, "takazawa_miaya", "sakura hanpen"),
    (1469320, 692, 723, "hatsushiba_chitose", "ichiri"),
    (1469320, 752, 820, "kamikawa_saya", "sakura hanpen"),
    (1469320, 821, 898, "takazawa_miaya", "sakura hanpen"),
    
    # Princess Evangile
    (393536, 2, 100, "rousen'in_rise", "yamakaze ran"),
    (393536, 101, 205, "sagisawa_chiho", "saeki nao"),
    (393536, 206, 312, "kitamikado_ritsuko", "saeki nao"),
    (393536, 313, 422, "kitamikado_ayaka", "yamakaze ran"),
    (393536, 445, 454, "myougi_marika", "yamakaze ran"),

    (800236, 4, 99, "rousen'in_rise", "yamakaze ran"),
    (800236, 100, 204, "sagisawa_chiho", "saeki nao"),
    (800236, 205, 311, "kitamikado_ritsuko", "saeki nao"),
    (800236, 312, 421, "kitamikado_ayaka", "yamakaze ran"),
    (800236, 444, 453, "myougi_marika", "yamakaze ran"),
    (800236, 454, 461, "rousen'in_rise", "yamakaze ran"),
    (800236, 462, 466, "kitamikado_ritsuko", "saeki nao"),
    (800236, 467, 468, "kitamikado_ayaka", "yamakaze ran"),
    (800236, 469, 472, "sagisawa_chiho", "saeki nao"),

    (3467979, 1, 219, "rousen'in_rise", "yamakaze ran"),
    (3467979, 220, 443, "sagisawa_chiho", "saeki nao"),
    (3467979, 444, 673, "kitamikado_ritsuko", "saeki nao"),
    (3467979, 674, 921, "kitamikado_ayaka", "yamakaze ran"),
    (3467979, 922, 1075, "kamiyagi_ruriko", "yamakaze ran"),
    (3467979, 1256, 1303, "mekata_mitsuki", "yamakaze ran"),
    (3467979, 1304, 1363, "nogi_tamie", "saeki nao"),
    (3467979, 1364, 1402, "myougi_marika", "yamakaze ran"),

    (1093800, 3, 60, "rousen'in_rise", "yamakaze ran"),
    (1093800, 61, 118, "sagisawa_chiho", "saeki nao"),
    (1093800, 119, 174, "kitamikado_ritsuko", "saeki nao"),
    (1093800, 175, 226, "kitamikado_ayaka", "yamakaze ran"),
    (1093800, 234, 295, "kamiyagi_ruriko", "yamakaze ran"),
    (1093800, 373, 434, "mekata_mitsuki", "yamakaze ran"),
    (1093800, 435, 497, "nogi_tamie", "saeki nao"),
    (1093800, 511, 603, "myougi_marika", "yamakaze ran"),

    (1329062, 2, 59, "rousen'in_rise", "yamakaze ran"),
    (1329062, 60, 117, "sagisawa_chiho", "saeki nao"),
    (1329062, 118, 173, "kitamikado_ritsuko", "saeki nao"),
    (1329062, 174, 225, "kitamikado_ayaka", "yamakaze ran"),
    (1329062, 233, 294, "kamiyagi_ruriko", "yamakaze ran"),
    (1329062, 372, 433, "mekata_mitsuki", "yamakaze ran"),
    (1329062, 434, 496, "nogi_tamie", "saeki nao"),
    (1329062, 506, 598, "myougi_marika", "yamakaze ran"),

    (3467980, 1, 28, "sagisawa_chiho", "saeki nao"),
    (3467980, 29, 51, "kitamikado_ritsuko", "saeki nao"),
    (3467980, 52, 113, "kitamikado_ayaka", "yamakaze ran"),
    (3467980, 114, 433, "kamiyagi_ruriko", "yamakaze ran"),
    (3467980, 590, 757, "mekata_mitsuki", "yamakaze ran"),
    (3467980, 758, 960, "nogi_tamie", "saeki nao"),
    (3467980, 961, 1098, "myougi_marika", "yamakaze ran"),
    # Toriko no Shimai
    (1944572, 3, 3, "uryuu_futaba", "teeta.j"),
    (1944572, 362, 494, "uryuu_futaba", "teeta.j"),
    (1944572, 495, 630, "yurimoto_yuna", "teeta.j"),

    (1945558, 358, 490, "uryuu_futaba", "teeta.j"),
    (1945558, 491, 510, "yurimoto_yuna", "teeta.j"),

    (1947184, 26, 33, "uryuu_futaba", "teeta.j"),
    (1947184, 42, 48, "uryuu_futaba", "teeta.j"),
    (1947184, 72, 76, "uryuu_futaba", "teeta.j"),

    (2555429, 25, 32, "uryuu_futaba", "teeta.j"),
    (2555429, 41, 47, "uryuu_futaba", "teeta.j"),
    (2555429, 87, 91, "uryuu_futaba", "teeta.j"),

    (2311852, 541, 732, "uryuu_futaba", "teeta.j"),
    # Ore no Ue de Agaku Rokunin no Togime
    (1328609, 1, 135, "amaya_tomoko", "noba"),
    (1328609, 136, 249, "kaji_nana", "noba"),
    (1328609, 250, 376, "orikura_satsuki", "mizuyuki"),
    (1328609, 377, 491, "suou_risa", "mizuyuki"),
    (1328609, 492, 600, "tsukino_yuri", "teeta.j"),
    (1328609, 601, 712, "ruka_(ore_no_ue_de_agaku_rokunin_no_togime)", "teeta.j"),
    (1329539, 1, 136, "amaya_tomoko", "noba"),
    (1329539, 137, 250, "kaji_nana", "noba"),
    (1329539, 251, 369, "orikura_satsuki", "mizuyuki"),
    (1329539, 370, 484, "suou_risa", "mizuyuki"),
    (1329539, 485, 593, "tsukino_yuri", "teeta.j"),
    (1329539, 594, 705, "ruka_(ore_no_ue_de_agaku_rokunin_no_togime)", "teeta.j"),
    # muutsuki
    # Hoshizora e Kakaru Hashi
    (485439, 11, 47, None, "naturalton"),
    (485439, 48, 77, "toudou_kasane", "arikawa satoru"),
    (485439, 78, 90, None, "ryohka"),
    (485439, 91, 156, "koumoto_madoka", "ryohka"),
    (485439, 157, 188, "nagase_minato", "ryohka"),
    (485439, 189, 421, "nanamori_seira", "asaba yuu"),
    (485439, 422, 540, "yorozu_senka", "arikawa satoru"),
    (485439, 541, 561, "toudou_tsumugi", "tsurusaki takahiro"),
    (485439, 562, 582, "nakatsugawa_ui", "ryohka"),
    (485439, 583, 683, None, "naturalton"),
    (485439, 684, 790, "yorozu_senka", "arikawa satoru"),
    (485439, 791, 811, "toudou_tsumugi", "tsurusaki takahiro"),
    (485439, 812, 832, "nakatsugawa_ui", "ryohka"),

    (296305, 13, 146, None, "naturalton"),
    (296305, 147, 205, None, "ryohka"),
    (296305, 206, 303, "koumoto_madoka", "ryohka"),
    (296305, 307, 366, "toudou_tsumugi", "tsurusaki takahiro"),
    (296305, 367, 436, "nakatsugawa_ui", "ryohka"),

    (577648, 13, 146, None, "naturalton"),
    (577648, 147, 205, None, "ryohka"),
    (577648, 206, 303, "koumoto_madoka", "ryohka"),
    (577648, 307, 367, "toudou_tsumugi", "tsurusaki takahiro"),
    (577648, 368, 440, "nakatsugawa_ui", "ryohka"),

    (1432111, 14, 147, None, "naturalton"),
    (1432111, 148, 206, None, "ryohka"),
    (1432111, 207, 304, "koumoto_madoka", "ryohka"),
    (1432111, 308, 368, "toudou_tsumugi", "tsurusaki takahiro"),
    (1432111, 369, 438, "nakatsugawa_ui", "ryohka"),
    (1432111, 439, 457, None, "naturalton"),
    (1432111, 461, 481, "nakatsugawa_ui", "ryohka"),

    (1432112, 13, 49, None, "naturalton"),
    (1432112, 50, 79, "toudou_kasane", "arikawa satoru"),
    (1432112, 80, 92, None, "ryohka"),
    (1432112, 93, 158, "koumoto_madoka", "ryohka"),
    (1432112, 159, 190, "nagase_minato", "ryohka"),
    (1432112, 191, 423, "nanamori_seira", "asaba yuu"),
    (1432112, 424, 542, "yorozu_senka", "arikawa satoru"),
    (1432112, 543, 563, "toudou_tsumugi", "tsurusaki takahiro"),
    (1432112, 564, 584, "nakatsugawa_ui", "ryohka"),
    (1432112, 585, 685, None, "naturalton"),

    # Osananajimi no Onee-chan Sensei to H de Naisho na Kankei!?
    (1482351, 1, 2000, "yokura_kitaka", "azuki yui"),

    # Nagaruboshi
    (3019741, 1, 2000, "mikoto_(nagaruboshi)", "nanaca mai"),
    (3498433, 1, 77, "mikoto_(nagaruboshi)", "nanaca mai"),
    # Loca Love
    (1741695, 1, 2000, "shizuki_yachiyo", "nanaca mai"),
    (1741050, 1, 2000, "aritagawa_nio", "nanaca mai"),
    (1741023, 1, 2000, "aritagawa_nio", "nanaca mai"),
    (1740957, 1, 2000, "shizuki_yachiyo", "nanaca mai"),
    (1476475, 1, 2000, "aritagawa_nio", "nanaca mai"),
    (1306942, 1, 2000, "kojika_hiwa", "nanaca mai"),
    # Kami no Ue no Mahoutsukai
    (767743, 2, 7, "yuugyouji_yoruko", "kiriha"),
    (767743, 18, 19, "yuugyouji_yoruko", "kiriha"),
    (767743, 48, 50, "yuugyouji_yoruko", "kiriha"),
    (767743, 121, 136, "yuugyouji_yoruko", "kiriha"),
    (767743, 247, 268, "yuugyouji_yoruko", "kiriha"),
    (767743, 288, 292, "yuugyouji_yoruko", "kiriha"),
    (767743, 322, 326, "yuugyouji_yoruko", "kiriha"),
    (767743, 329, 332, "yuugyouji_yoruko", "kiriha"),
    (767743, 430, 461, "yuugyouji_yoruko", "kiriha"),

    (767743, 13, 17, "fushimi_rio", "kiriha"),
    (767743, 43, 47, "fushimi_rio", "kiriha"),
    (767743, 51, 95, "fushimi_rio", "kiriha"),
    (767743, 100, 102, "fushimi_rio", "kiriha"),
    (767743, 109, 112, "fushimi_rio", "kiriha"),
    (767743, 183, 186, "fushimi_rio", "kiriha"),
    (767743, 269, 273, "fushimi_rio", "kiriha"),
    (767743, 297, 303, "fushimi_rio", "kiriha"),

    (767743, 8, 12, "himukai_kanata", "kiriha"),
    (767743, 20, 24, "himukai_kanata", "kiriha"),
    (767743, 37, 42, "himukai_kanata", "kiriha"),
    (767743, 96, 99, "himukai_kanata", "kiriha"),
    (767743, 113, 120, "himukai_kanata", "kiriha"),
    (767743, 161, 176, "himukai_kanata", "kiriha"),
    (767743, 311, 321, "himukai_kanata", "kiriha"),
    (767743, 336, 341, "himukai_kanata", "kiriha"),
    (767743, 394, 429, "himukai_kanata", "kiriha"),

    (767743, 103, 108, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 137, 155, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 177, 182, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 190, 246, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 293, 296, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 327, 328, "tsukiyashiro_kisaki", "kiriha"),
    (767743, 342, 347, "tsukiyashiro_kisaki", "kiriha"),

    # Unlucky Re：Birth/Reverse
    (868699, 253, 550, "eris_elenoare", "nanase meruchi"),
    (868699, 551, 630, "mirina_liliano", "miyasu risa"),
    (868699, 631, 697, "aria_celestia", "nanase meruchi"),
    (868699, 698, 879, "lisley_mcdowell", "miyasu risa"),
    (868699, 891, 898, "lisley_mcdowell", "miyasu risa"),

    (868699, 880, 1848, "eris_elenoare", "nanase meruchi"),
    (868699, 1849, 2000, "mirina_liliano", "miyasu risa"),
    (868706, 1, 192, "mirina_liliano", "miyasu risa"),
    (868706, 193, 641, "aria_celestia", "nanase meruchi"),
    (868706, 642, 976, "lisley_mcdowell", "miyasu risa"),

    # Inochi no Spare
    (3189442, 103, 104, "shukugawa_meguri", "akizora momiji"),
    (3189442, 166, 341, "shukugawa_meguri", "akizora momiji"),
    (969417, 1, 3, "shukugawa_meguri", "akizora momiji"),
    (969417, 24, 65, "shukugawa_meguri", "akizora momiji"),
    (969417, 68, 86, "shukugawa_meguri", "akizora momiji"),
    (969417, 109, 244, "shukugawa_meguri", "akizora momiji"),
    (1420982, 1, 2000, "shukugawa_meguri", "akizora momiji"),
    (1420992, 1, 1195, "shukugawa_meguri", "akizora momiji"),

    # Teaka Mamire no Tenshi
    (951022, 1, 275, "kisaragi_reina", "akizora momiji"),
    (951022, 276, 288, "anri_(teaka_mamire_no_tenshi)", "akizora momiji"),
    (1375832, 1, 600, "kisaragi_reina", "akizora momiji"),
    (1375832, 601, 696, "anri_(teaka_mamire_no_tenshi)", "akizora momiji"),
    # Dekinai Watashi ga, Kurikaesu
    (733733, 305, 314, "kurihara_yume", "aotsuki shinobu"),
    (733733, 321, 324, "kurihara_yume", "aotsuki shinobu"),
    (733733, 332, 384, "kurihara_yume", "aotsuki shinobu"),
    # Aikagi
    (1919732, 1, 2000, "saotome_ai", "gintarou"),
    (3442621, 1, 2000, "saotome_ai", "gintarou"),
    (1403914, 1, 2000, "sumeragi_ayano", "gintarou"),
    (3442620, 1, 2000, "sumeragi_ayano", "gintarou"),
    (1021562, 1, 2000, "takanashi_shiori", "gintarou"),
    (2685512, 1, 2000, "takanashi_shiori", "gintarou"),
    (1175436, 1, 2000, "takanashi_shiori", "gintarou"),
    # Tensei Kunitori Sex Gassen!!
    (753612, 58, 160, None, "minakami rinka"),
    (753612, 161, 426, "ouma_mizuki", "2-g"),
    (753612, 427, 712, "hozumi_moa", "2-g"),
    (753612, 713, 1112, "tsukishiro_nami", None),
    (753612, 1113, 1488, "komae_nana", "2-g"),
    (753612, 1489, 1755, "hitotonoya_yuuri", "sukoyaka gyuunyuu"),

    (1404939, 2, 171, "ouma_mizuki", "2-g"),
    (1404939, 172, 442, "hozumi_moa", "2-g"),
    (1404939, 443, 517, "hitotonoya_yuuri", "sukoyaka gyuunyuu"),
    (1404939, 518, 778, "komae_nana", "2-g"),
    # Momoiro Seiheki Kaihou Sengen!
    (691358, 2, 98, "inamori_chiduru", "ichiyo moka"),
    (691358, 99, 283, "oda_mao", "2-g"),
    (691358, 284, 572, "kiryuuin_rindou", "annie"),
    (691358, 573, 628, "ooe_kagura", "minakami rinka"),
    (691358, 629, 760, "kitami_karen", "sukoyaka gyuunyuu"),

    (691359, 15, 79, "inamori_chiduru", "ichiyo moka"),
    (691359, 80, 261, "oda_mao", "2-g"),
    (691359, 262, 380, "kiryuuin_rindou", "annie"),
    (691359, 396, 449, "kiryuuin_rindou", "annie"),
    (691359, 450, 477, "ooe_kagura", "minakami rinka"),
    (691359, 478, 565, "kitami_karen", "sukoyaka gyuunyuu"),

    (691357, 2, 291, "inamori_chiduru", "ichiyo moka"),
    (691357, 816, 1171, "oda_mao", "2-g"),
    (691357, 1259, 1589, "kiryuuin_rindou", "annie"),
    (691357, 292, 587, "ooe_kagura", "minakami rinka"),
    (691357, 588, 815, "kitami_karen", "sukoyaka gyuunyuu"),

    # Maid in Witch Life
    (1356727, 1, 609, "alisa_forerulozzo", "ichiyo moka"),
    (1356727, 1314, 1714, "alisa_forerulozzo", "ichiyo moka"),
    (1356751, 42, 449, "liliana_echsun", "umetori uriri"),
    (1353705, 3, 259, "alisa_forerulozzo", "ichiyo moka"),
    (1353705, 535, 788, "liliana_echsun", "umetori uriri"),

    # Houkago⇒Education! ~Sensei to Hajimeru Miwaku no Lesson~
    (1964980, 2, 250, "tenkawa_sayuki", " "),
    (1964980, 443, 632, "orime_tamaki", None),

    # Sennagi
    (2997104, 1, 1650, "mikosono_himeka", "chobipero"),
    (2997104, 1651, 1758, "el_(sennagi)", "aiu"),
    (2997104, 1759, 2000, "tharja_(sennagi)", None),
    (2997105, 1, 58, "tharja_(sennagi)", None),
    (2997105, 59, 458, "mutsu_(sennagi)", None),
    (2997105, 459, 602, "charlotte_(sennagi)", None),
    (2997102, 1, 1882, "mikosono_himeka", "chobipero"),
    (2997102, 1896, 2000, "el_(sennagi)", "aiu"),
    (2720188, 1, 885, "mikosono_himeka", "chobipero"),
    (2720188, 886, 1018, "el_(sennagi)", "aiu"),
    (2720188, 1019, 1155, "tharja_(sennagi)", None),
    (2720188, 1156, 1282, "mutsu_(sennagi)", None),
    (2720188, 1283, 1373, "charlotte_(sennagi)", None),
    (2997103, 1, 57, "el_(sennagi)", "aiu"),
    (2997103, 58, 350, "tharja_(sennagi)", None),
    (2997103, 351, 568, "mutsu_(sennagi)", None),
    (2997103, 568, 736, "charlotte_(sennagi)", None),

    # Haison Shoujo
    (2710683, 1, 90, "kagome", "aiu"),
    (2710683, 91, 206, "yakumo_azusa", "yuurin"),
    (2710683, 207, 272, "kagami_shuri", "chobipero"),
    (2710683, 275, 344, "emma_aaron_yakushiin", "aose"),
    (2710683, 346, 432, "karasuno_tsubame", " "),
    (2710683, 435, 518, "furube_yurara", " "),
    (2710683, 521, 589, "osakabe_rei", " "),
    (2710683, 592, 660, "yuzuriha_manaka", " "),

    (2410167, 10, 183, "kagome", "aiu"),
    (2410167, 413, 660, "yakumo_azusa", "yuurin"),
    (2410167, 184, 412, "kagami_shuri", "chobipero"),
    (2410167, 661, 770, "emma_aaron_yakushiin", "aose"),
    (2410167, 771, 881, "karasuno_tsubame", " "),
    (2410167, 882, 988, "furube_yurara", " "),
    (2410167, 989, 1105, "osakabe_rei", " "),
    (2410167, 1106, 1208, "yuzuriha_manaka", " "),

    (3103887, 1, 174, "kagome", None),
    (3103887, 175, 388, "yakumo_azusa", None),
    (3103887, 389, 614, "kagami_shuri", None),
    (3103887, 615, 761, "emma_aaron_yakushiin", None),
    (3103887, 762, 872, "karasuno_tsubame", None),
    (3103887, 873, 979, "furube_yurara", None),
    (3103887, 980, 1096, "osakabe_rei", None),
    (3103887, 1097, 1199, "yuzuriha_manaka", None),
    # Oniichan migite no shiyou wo kinshi shimasu!
    (1179865, 13, 250, "imoo_tsugumi", "k-ko"),
    (1179865, 251, 262, "imoo_tsugumi, imoo_ayuka", "k-ko"),
    (1179865, 263, 510, "imoo_ayuka", "k-ko"),
    (1179865, 511, 543, "imoo_kaede, imoo_yuki", "hisama kumako"),
    (1179865, 544, 766, "imoo_kaede", "hisama kumako"),
    (1179865, 767, 783, "imoo_kaede, imoo_yuki", "hisama kumako"),
    (1179865, 784, 804, "imoo_tsugumi, imoo_ayuka", "k-ko"),
    (1179865, 805, 1018, "imoo_yuki", "hisama kumako"),
    (1179865, 1019, 1054, "imoo_tsugumi, imoo_ayuka, imoo_kaede, imoo_yuki", "hisama kumako, k-ko"),

    (1429557, 1, 823, "imoo_ayuka", "k-ko"),
    (1429557, 824, 1222, "imoo_kaede", "hisama kumako"),
    (1429564, 216, 830, "imoo_tsugumi", "k-ko"),
    (1429564, 831, 1448, "imoo_yuki", "hisama kumako"),

    (742906, 2, 230, "imoo_tsugumi", "k-ko"),
    (742906, 231, 459, "imoo_ayuka", "k-ko"),
    (742906, 460, 673, "imoo_kaede", "hisama kumako"),
    (742906, 674, 868, "imoo_yuki", "hisama kumako"),
    # Onii-chan Sharing
    (633556, 4, 121, "kurosu_seseri", "k-ko"),
    (633556, 133, 263, "nadeshiko_futaba", "k-ko"),
    (633556, 284, 394, "yotsunoha_chiko", "hisama kumako"),
    (633556, 402, 532, "yotsunoha_yuu", "hisama kumako"),
    # Onii-chan Teacher ~Himitsu no Jugyou o Kibou Shimasu!!~
    (887147, 29, 187, "kitami_kanae", "k-ko"),
    (887147, 188, 335, "kitami_ruri", "k-ko"),
    (887147, 336, 488, "kitami_ai", "pikazo"),
    (887147, 489, 634, "kitami_tsubomi", "pikazo"),

    (887147, 710, 951, "kitami_ai", "pikazo"),
    (887147, 952, 1191, "kitami_kanae", "k-ko"),
    (887147, 1192, 1443, "kitami_ruri", "k-ko"),
    (887147, 1444, 1652, "kitami_tsubomi", "pikazo"),
    # Chiccha na Hanayome 
    (1068104, 2, 146, "sakihara_rin", "usume shirou"),
    (1068104, 147, 276, "kounogi_risa", "usume shirou"),
    (1068104, 277, 290, "sakihara_rin, kounogi_risa", "usume shirou"),
    (1068104, 291, 394, "aoino_haru", "pikazo"),
    (1068104, 395, 500, "akieda_mashiro", "pikazo"),


    # Haison Shoujo [Ni]
    (3455889, 221, 268, "shinju, mishio_haruna", "takashina asahi"),
    (3455889, 280, 298, "shinju, mishio_haruna", "takashina asahi"),
    (3455889, 2, 295, "shinju", "takashina asahi"),
    (3455889, 299, 408, "mishio_haruna", "takashina asahi"),
    (3455889, 409, 433, "shinju, mishio_haruna", "takashina asahi"),
    (3455889, 434, 504, "mishio_haruna", "takashina asahi"),
    (3455889, 505, 556, "shinju, mishio_haruna", "takashina asahi"),
    (3455889, 557, 782, "ryuugatou_yachiyo", None),
    (3455889, 783, 1004, "niinuma_suzu", None),
    (3455889, 1005, 1117, "tsuduki_nanase", None),


    (3455917, 1, 243, "shinju", "takashina asahi"),
    (3455917, 244, 663, "mishio_haruna", "takashina asahi"),
    (3455917, 664, 845, "ryuugatou_yachiyo", None),
    (3455917, 846, 1160, "niinuma_suzu", None),
    (3455917, 1161, 1376, "tsuduki_nanase", None),

    (3455916, 1, 243, "shinju", "takashina asahi"),
    (3455916, 244, 663, "mishio_haruna", "takashina asahi"),
    (3455916, 664, 845, "ryuugatou_yachiyo", None),
    (3455916, 846, 1160, "niinuma_suzu", None),
    (3455916, 1161, 1376, "tsuduki_nanase", None),

    # Watashi ga Suki nara "Suki" tte Itte
    (869167, 4, 397, "himekami_ayame", "chiri"),
    (869167, 399, 756, "goshogawara_yuuki", "k-ko"),
    (869167, 757, 1226, "komachi_mahiru", "mango pudding"),
    (869167, 1227, 1386, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", "mango pudding"),
    (869167, 1393, 1589, "yataka_chiho", "syroh"),

    (1328827, 3, 398, "himekami_ayame", "chiri"),
    (1328827, 399, 776, "goshogawara_yuuki", "k-ko"),
    (1328827, 777, 1247, "komachi_mahiru", "mango pudding"),
    (1328827, 1248, 1416, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", "mango pudding"),
    (1328827, 1423, 1629, "yataka_chiho", "syroh"),

    (1959114, 1, 2000, "himekami_ayame", None),
    (1959111, 1, 2000, "himekami_ayame", None),
    (1959096, 1, 2000, "himekami_ayame", None),
    (1959088, 1, 2000, "himekami_ayame", None),
    (1959086, 1, 2000, "goshogawara_yuuki", None),
    (1959079, 1, 2000, "goshogawara_yuuki", None),
    (1959072, 1, 2000, "goshogawara_yuuki", None),
    (1959067, 1, 2000, "goshogawara_yuuki", None),
    (1959050, 1, 2000, "komachi_mahiru", None),
    (1959040, 1, 2000, "komachi_mahiru", None),
    (1959035, 1, 2000, "komachi_mahiru", None),
    (1959027, 1, 2000, "komachi_mahiru", None),
    (1959145, 1, 2000, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", None),
    (1959133, 1, 2000, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", None),
    (1959124, 1, 2000, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", None),
    (1959118, 1, 2000, "rinka_(watashi_ga_suki_nara_\"suki\"_tte_itte!)", None),
    (1959062, 1, 2000, "yataka_chiho", None),
    (1959053, 1, 2000, "yataka_chiho", None),
    # soi_kano_~gyutto_dakishimete~
    (1175726, 3, 127, "hanatsuka_aika", "ameto yuki"),
    (1175726, 157, 329, "chiyuki_touko", "hinata momo"),
    (1175726, 330, 498, "katagiri_tsubame", "kurasawa moko"),
    (1175726, 499, 648, "kumakura_yoake", "hisama kumako"),

    (3428411, 1, 330, "hanatsuka_aika", "ameto yuki"),
    (3428411, 508, 867, "chiyuki_touko", "hinata momo"),
    (3428411, 906, 1499, "katagiri_tsubame", "kurasawa moko"),
    (3428412, 1, 798, "kumakura_yoake", "hisama kumako"),
    # onii-chan_kiss_no_junbi_wa_mada_desu_ka?
    (929330, 3, 292, "seguchi_asahi", "k-ko"),
    (929330, 293, 608, "seguchi_mahiru", "k-ko"),
    (929330, 609, 897, "seguchi_yayoi", "sakura misaki"),
    (929330, 898, 1175, "seguchi_saya", "sakura misaki"),

    (1488369, 3, 294, "seguchi_asahi", "k-ko"),
    (1488369, 295, 610, "seguchi_mahiru", "k-ko"),
    (1488369, 611, 900, "seguchi_yayoi", "sakura misaki"),
    (1488369, 901, 1178, "seguchi_saya", "sakura misaki"),

    (1035261, 2, 65, "seguchi_asahi", "k-ko"),
    (1035261, 66, 122, "seguchi_mahiru", "k-ko"),
    (1035261, 123, 171, "seguchi_yayoi", "sakura misaki"),
    (1035261, 172, 229, "seguchi_saya", "sakura misaki"),
    (1035261, 233, 362, "seguchi_asahi", "k-ko"),
    (1035261, 363, 454, "seguchi_mahiru", "k-ko"),
    (1035261, 455, 568, "seguchi_yayoi", "sakura misaki"),
    (1035261, 569, 654, "seguchi_saya", "sakura misaki"),

    (1370031, 2, 290, "seguchi_asahi", "k-ko"),
    (1370031, 291, 580, "seguchi_mahiru", "k-ko"),
    (1370031, 581, 893, "seguchi_yayoi", "sakura misaki"),
    (1370031, 894, 1170, "seguchi_saya", "sakura misaki"),

    (1057065, 1, 2000, "seguchi_youkou", "sakura misaki"),

    # uso(campus)
    (878564, 1, 2000, "himeno_satsuki", "riichu"),
    (928204, 1, 2000, "izumi_aoi", "riichu"),
    (998646, 1, 2000, "eris_fall_cartlet", "riichu"),
    (1096716, 1, 2000, "teidou_setsuka", "riichu"),

    (1209328, 64, 89, "himeno_satsuki", "riichu"),
    (1209328, 5, 31, "izumi_aoi", "riichu"),
    (1209328, 32, 57, "eris_fall_cartlet", "riichu"),
    (1209328, 90, 127, "teidou_setsuka", "riichu"),

    # Kanojo no Seiiki
    (2493223, 1, 2000, "akiyoshi_fuyuka", "ryohka"),
    (769445, 1, 2000, "akiyoshi_fuyuka", "ryohka"),
    (847646, 1, 136, "nase_yukana", "ryohka"),
    (2593173, 12, 163, "nase_yukana", "ryohka"),
    (1000114, 5, 116, "ootori_maika", "ryohka"),
    (1000114, 117, 231, "nase_yaeka", "ryohka"),
    (2771291, 5, 116, "ootori_maika", "ryohka"),
    (2771291, 117, 231, "nase_yaeka", "ryohka"),

    # Hanahime * Absolute!
    (1414902, 1, 305, "nekoyashiki_mea", "kannagi rei"),
    (1414902, 306, 513, "reina_lil_rafen", "kannagi rei"),
    (1414902, 514, 686, "polina_mirova_von_schwarzacher", "kannagi rei"),
    (1414902, 687, 873, "aiuchi_hiyoko", "kannagi rei"),
    (1414902, 874, 1007, "kamiya_ibu", "kannagi rei"),
    (1414917, 1, 193, "ootori_anko", "kannagi rei"),
    (1414917, 194, 421, "kumanomidou_ayane", "kannagi rei"),
    (1414917, 422, 459, "biizu-chan", "kannagi rei"),
    (1414917, 842, 861, "sawa", "tanihara natsuki"),
    (970633, 5, 60, "nekoyashiki_mea", "kannagi rei"),
    (970633, 61, 120, "reina_lil_rafen", "kannagi rei"),
    (970633, 121, 175, "polina_mirova_von_schwarzacher", "kannagi rei"),
    (970633, 176, 226, "aiuchi_hiyoko", "kannagi rei"),
    (970633, 227, 260, "kamiya_ibu", "kannagi rei"),
    (970633, 261, 265, "ootori_anko", "kannagi rei"),
    (970633, 266, 269, "kumanomidou_ayane", "kannagi rei"),
    # Shukusei no Girlfriend
    (1262323, 1, 2000, "kamiizumi_yuuri", "kannagi rei"),
    (1333093, 1, 2000, "maaya_klienke", "kannagi rei"),
    (1333084, 1, 2000, "maaya_klienke", "kannagi rei"),
    (1453992, 1, 2000, "satake_kanoko", "kannagi rei"),
    (2265190, 1, 2000, "yonamine_fujiko", "kannagi rei"),
    # Mayoeru Futari to Sekai no Subete Love Heaven 300%
    (821044, 21, 255, "mizunomiya_nana", "moriyama shijimi"),
    (821044, 256, 433, "saeki_touka", "youta"),
    (821044, 434, 474, "fia", "youta"),

    (758201, 60, 135, "mizunomiya_nana", "moriyama shijimi"),
    (758201, 136, 172, "hirohara_nayuta", "moriyama shijimi"),
    (758201, 173, 226, "hasekura_otoha", "moriyama shijimi"),
    (758201, 227, 273, "yuuki_suzuran", "youta"),
    (758201, 274, 342, "saeki_touka", "youta"),
    (758201, 352, 420, "fia", "youta"),
    # Midare Setsugetsuka
    (877937, 58, 172, "sasanome_yukina", "minatsuki alumi"),
    (877937, 173, 247, "nayotake_mitsuki", "moriyama shijimi"),
    (877937, 248, 371, "sasanome_yukina", "minatsuki alumi"),
    (877937, 396, 429, "nayotake_mitsuki", "moriyama shijimi"),

    (969692, 4, 11, "sasanome_yukina", "minatsuki alumi"),
    (969692, 28, 52, "mamiya_rurika", "moriyama shijimi"),
    (969692, 53, 129, "nayotake_mitsuki", "moriyama shijimi"),
    (969692, 130, 148, "sasanome_yukina", "minatsuki alumi"),
    (969692, 149, 151, "mamiya_rurika", "moriyama shijimi"),
    (969692, 162, 211, "nayotake_mitsuki", "moriyama shijimi"),
    (969692, 212, 230, "sasanome_yukina", "minatsuki alumi"),
    (969692, 231, 241, "nayotake_mitsuki", "moriyama shijimi"),
    (969692, 250, 264, "sasanome_yukina", "minatsuki alumi"),
    (969692, 265, 295, "umino_otome", "moriyama shijimi"),
    (969692, 296, 319, "nayotake_mitsuki", "moriyama shijimi"),
    (969692, 320, 352, "mamiya_rurika", "moriyama shijimi"),
    (969692, 353, 379, "nayotake_mitsuki", "moriyama shijimi"),
    (969692, 380, 407, "umino_otome", "moriyama shijimi"),
    (969692, 408, 423, "mamiya_rurika", "moriyama shijimi"),
    (969692, 424, 452, "sasanome_yukina", "minatsuki alumi"),
    (969692, 453, 486, "mamiya_rurika", "moriyama shijimi"),
    (969692, 487, 509, "sasanome_yukina", "minatsuki alumi"),
    (969692, 527, 537, "sasanome_yukina", "minatsuki alumi"),
    (969692, 538, 552, "umino_otome", "moriyama shijimi"),
    (969692, 554, 591, "mamiya_rurika", "moriyama shijimi"),
    (969692, 592, 605, "sasanome_yukina", "minatsuki alumi"),
    (969692, 606, 632, "nayotake_mitsuki", "moriyama shijimi"),

    (1397185, 1, 888, "nayotake_mitsuki", "moriyama shijimi"),
    (1397185, 1687, 2000, "sasanome_yukina", "minatsuki alumi"),
    (1397257, 1, 346, "sasanome_yukina", "minatsuki alumi"),
    (1397257, 547, 622, "umino_otome", "moriyama shijimi"),
    (1397257, 727, 1006, "mamiya_rurika", "moriyama shijimi"),
    # Pure Marriage
    (938497, 1, 180, "hoshigaoka_madoka", "furukawa remon"),
    (938497, 215, 284, "hoshigaoka_madoka", "furukawa remon"),

    (1133863, 1, 141, "hirohara_d_serika", "furukawa remon"),
    (1133863, 142, 154, "qliphoth", "moriyama shijimi"),
    (1133863, 155, 170, "hirohara_d_serika", "furukawa remon"),

    (1158938, 56, 104, "hanasaki_sakura", "moriyama shijimi"),
    (1158938, 193, 445, "hanasaki_sakura", "moriyama shijimi"),

    (1205421, 25, 46, "hoshigaoka_madoka", "furukawa remon"),
    (1205421, 47, 87, "hanasaki_sakura", "moriyama shijimi"),
    (1205421, 110, 145, "hoshigaoka_madoka", "furukawa remon"),
    (1205421, 156, 159, "hanasaki_sakura", "moriyama shijimi"),
    (1205421, 160, 188, "hoshigaoka_madoka", "furukawa remon"),
    (1205421, 189, 215, "hirohara_d_serika", "furukawa remon"),
    (1205421, 216, 224, "hanasaki_sakura", "moriyama shijimi"),
    (1205421, 225, 232, "qliphoth", "moriyama shijimi"),
    (1205421, 302, 339, "qliphoth", "moriyama shijimi"),
    (1205421, 340, 386, "hoshigaoka_madoka", "furukawa remon"),

    (1135078, 1, 320, "hanasaki_sakura", "moriyama shijimi"),
    (1135078, 321, 368, "hoshigaoka_madoka", "furukawa remon"),
    (1135078, 378, 395, "qliphoth", "moriyama shijimi"),
    (1135078, 396, 539, "hirohara_d_serika", "furukawa remon"),

    (3243128, 1, 36, "qliphoth", "moriyama shijimi"),
    # Liber_7 Eigou no Owari o Matsu Kimi e
    (1008861, 7, 284, "ichijou_kurea", "moriyama shijimi"),
    (1008861, 331, 528, "hirohara_mei", "youta"),
    (1008861, 529, 702, "amatsuka_miku", "kawai maria"),
    (1008861, 743, 892, "kusunose_saaya", "kawai maria"),
    # Kakeochi
    (1403993, 1, 96, "aizome_kaguya", "sasahiro"),
    (1413731, 1, 561, "aizome_kaguya", "sasahiro"),
    # Little Sick Girls
    (1491186, 1, 192, "yoshino_eri", "moriyama shijimi"),
    (1536973, 1, 129, "takaido_ruchie", "moriyama shijimi"),
    (1689666, 1, 146, "aiba_momo", "moriyama shijimi"),

    (1809449, 54, 209, "takaido_ruchie", "moriyama shijimi"),
    (1809449, 210, 339, "yoshino_eri", "moriyama shijimi"),
    (1809449, 340, 547, "aiba_momo", "moriyama shijimi"),

    # Konekone Koneko
    (1215915, 52, 186, "nekokawa_shirone", "noda shuha"),
    (1215915, 187, 188, "nekokawa_shirone, nekoya_kohina, nekohada_miyabi", "noda shuha, naenae, wori"),
    (1215915, 189, 332, "nekoya_kohina", "naenae"),
    (1215915, 333, 486, "nekohada_miyabi", "wori"),
    (1215915, 521, 555, "nekokawa_shirone, nekoya_kohina, nekohada_miyabi", "noda shuha, naenae, wori"),
    # Can Can Bunny Premiere 3
    (1305782, 7, 98, None, "wori"),
    (1305782, 465, 473, None, "wori"),
    (1305782, 310, 396, None, "naenae"),
    (1305782, 524, 545, None, "naenae"),
    (1305782, 205, 309, None, "rokudou itsuki"),
    (1305782, 504, 523, None, "rokudou itsuki"),
    (1305782, 99, 204, "suzaki_kagome", "noda shuha"),
    (1305782, 474, 503, "suzaki_kagome", "noda shuha"),
    # Hoshizora Tea Party Extra ~"Ai" Hajimarimashita!~
    (1043987, 577, 687, "yamane_nemu", "noda shuha"),
    (1043987, 1017, 1032, "yamane_nemu", "noda shuha"),
    (1043987, 261, 276, "arisuno_arisu", "wori"),
    (1043987, 316, 375, "arisuno_arisu", "wori"),
    (1043987, 962, 985, "arisuno_arisu", "wori"),
    (1043987, 376, 465, None, "rokudou itsuki"),
    (1043987, 986, 1005, None, "rokudou itsuki"),
    # Tsukumome Aoe
    (741954, 1, 141, "aoe", "kokusan moyashi"),
    # Tsukiyo no Mura
    (763324, 2, 58, "kawanaka_moeka, kamijou_shiori", "rozea"),
    (763324, 59, 142, "kawanaka_moeka", "rozea"),
    (763324, 143, 331, "kamijou_shiori", "rozea"),

    (1119923, 1, 156, "kawanaka_moeka", "rozea"),
    (1119923, 157, 268, "kamijou_shiori", "rozea"),
    # Koi iro Marriage
    (1182420, 5, 13, "morikawa_mihono", "sasorigatame"),
    (1182420, 19, 381, "takamiya_kuon", "murakami suigun"),
    (1182420, 384, 847, "morikawa_mihono", "sasorigatame"),
    (1182420, 848, 1200, "akiyoshi_nekoko", "sasorigatame"),
    (1182420, 1202, 1568, "luriastis_t_mikuriya", "chikotam"),
    # Mirai Kanojo
    (807638, 123, 590, "touno_kanae", "oota yuuichi"),

    (807638, 591, 642, "yuugiri_(mirai_kanojo)", "annie"),
    (807638, 643, 695, "kakiya_koyuki", "annie"),
    (807638, 696, 754, "yuugiri_(mirai_kanojo), kakiya_koyuki", "annie"),
    (807638, 755, 767, "kakiya_koyuki", "annie"),
    (807638, 768, 792, "yuugiri_(mirai_kanojo), kakiya_koyuki", "annie"),
    (807638, 793, 817, "yuugiri_(mirai_kanojo)", "annie"),
    (807638, 818, 975, "yuugiri_(mirai_kanojo), kakiya_koyuki", "annie"),

    (807638, 976, 1468, "sena_nonoka", "yuki makoto"),
    (807638, 1469, 1764, "toyota_yuzuki", "annie"),
    # Fureraba ~Friend to Lover~
    (607567, 2, 139, "minahara_himari", "ameto yuki"),
    (607567, 140, 321, "mochizuki_rina", "rei"),
    (607567, 322, 495, "hiiragi_yuzuyu", "hinata momo"),
    (607567, 498, 609, "sawatari_misaki", "ameto yuki"),

    (735501, 3, 21, "minahara_himari", "ameto yuki"),
    (735501, 22, 44, "mochizuki_rina", "rei"),
    (735501, 45, 57, "hiiragi_yuzuyu", "hinata momo"),
    (735501, 58, 81, "sawatari_misaki", "ameto yuki"),

    (989888, 3, 282, "minahara_himari", "ameto yuki"),
    (989888, 502, 795, "mochizuki_rina", "rei"),
    (989888, 796, 1100, "hiiragi_yuzuyu", "hinata momo"),
    (989888, 285, 501, "sawatari_misaki", "ameto yuki"),

    (994172, 3, 27, "minahara_himari", "ameto yuki"),
    (994172, 57, 88, "mochizuki_rina", "rei"),
    (994172, 89, 105, "hiiragi_yuzuyu", "hinata momo"),
    (994172, 28, 56, "sawatari_misaki", "ameto yuki"),

    (1206439, 2, 281, "minahara_himari", "ameto yuki"),
    (1206439, 501, 794, "mochizuki_rina", "rei"),
    (1206439, 795, 1099, "hiiragi_yuzuyu", "hinata momo"),
    (1206439, 284, 500, "sawatari_misaki", "ameto yuki"),

    (1316592, 2, 281, "minahara_himari", "ameto yuki"),
    (1316592, 501, 794, "mochizuki_rina", "rei"),
    (1316592, 795, 1099, "hiiragi_yuzuyu", "hinata momo"),
    (1316592, 284, 500, "sawatari_misaki", "ameto yuki"),

    (1461004, 3, 27, "minahara_himari", "ameto yuki"),
    (1461004, 57, 88, "mochizuki_rina", "rei"),
    (1461004, 89, 105, "hiiragi_yuzuyu", "hinata momo"),
    (1461004, 28, 56, "sawatari_misaki", "ameto yuki"),
    # Otome * Domain
    (948037, 6, 55, "saionji_kazari", "tatekawa mako"),
    (948037, 56, 119, "kifune_yuzu", "tatekawa mako"),
    (948037, 120, 173, "oogaki_hinata", "tatekawa mako"),
    (948037, 185, 256, "saionji_kazari", "tatekawa mako"),
    (948037, 257, 353, "kifune_yuzu", "tatekawa mako"),
    (948037, 354, 440, "oogaki_hinata", "tatekawa mako"),

    (3284505, 1, 2000, "saionji_kazari", "tatekawa mako"),
    (3284511, 1, 2000, "oogaki_hinata", "tatekawa mako"),
    (3284512, 1, 290, "oogaki_hinata", "tatekawa mako"),
    (3284512, 291, 2000, "kifune_yuzu", "tatekawa mako"),
    (3284513, 1, 2000, None, "tatekawa mako"),
    # Yakimochi Stream
    (742858, 2, 485, "kirishima_sana", "syroh"),
    (742858, 503, 979, "tania_helvellyn", "syroh"),
    (742858, 1006, 1456, "ibuki_kohane", "syroh"),
    (742858, 1472, 1898, "yukikura_mutsuki", "syroh"),
    # Anata o Otoko ni Shiteageru!
    (929570, 2, 129, "futaba_itsuki", "chiri"),
    (929570, 130, 321, "ayukawa_kogane", "syroh"),
    (929570, 322, 453, "toumi_beniyuki", "chiri"),
    (929570, 454, 627, "shuukaidou_rika", "syroh"),
    # Raspberry Cube
    (1293000, 4, 274, "kaidou_mikoto", "hasune"),
    (1293000, 305, 552, "kanou_minato", "hasune"),
    (1293000, 590, 826, "sakuraba_victoria_ruri", "hasune"),
    (1293000, 845, 1100, "yuzuki_yuu", "hasune"),
    # Tokeijikake no Ley-Line
    (513772, 126, 149, "shishigatani_ushio", "urabi"),
    (562008, 171, 205, "shishigatani_ushio", "urabi"),
    (562279, 171, 205, "shishigatani_ushio", "urabi"),
    (782270, 16, 31, "shishigatani_ushio", "urabi"),
    (782270, 150, 220, "shishigatani_ushio", "urabi"),
    (1139193, 126, 149, "shishigatani_ushio", "urabi"),
    (1336626, 164, 205, "shishigatani_ushio", "urabi"),
    # Kizuna Kirameku Koi Iroha
    (1334517, 2, 226, "kamiizumi_saya", "niro"),
    (1334517, 228, 398, "suzakuin_tsubaki", "pero"),
    (1334517, 399, 585, "aihara_shion", "usume shirou"),
    (1334517, 586, 746, "freesia_godspeed", "moeki yuuta"),

    (1354508, 408, 422, "kamiizumi_saya", "niro"),
    (1354508, 6, 29, "suzakuin_tsubaki", "pero"),
    (1354508, 48, 314, "suzakuin_tsubaki", "pero"),
    (1354508, 423, 437, "aihara_shion", "usume shirou"),
    (1354508, 438, 452, "freesia_godspeed", "moeki yuuta"),
    (1354508, 453, 485, "kamiizumi_saya, suzakuin_tsubaki, aihara_shion, freesia_godspeed", "niro, pero, usume shirou, moeki yuuta"),

    (1359096, 408, 422, "kamiizumi_saya", "niro"),
    (1359096, 6, 29, "suzakuin_tsubaki", "pero"),
    (1359096, 48, 314, "suzakuin_tsubaki", "pero"),
    (1359096, 423, 437, "aihara_shion", "usume shirou"),
    (1359096, 438, 452, "freesia_godspeed", "moeki yuuta"),
    (1359096, 453, 485, "kamiizumi_saya, suzakuin_tsubaki, aihara_shion, freesia_godspeed", "niro, pero, usume shirou, moeki yuuta"),

    (1696272, 1, 123, "suzakuin_tsubaki", "pero"),
    (1696272, 129, 2000, "suzakuin_tsubaki", "pero"),
    (1696276, 1, 2000, "aihara_shion", "usume shirou"),
    # Shiraha Kirameku Koi Shirabe
    (1525363, 45, 74, "kazamine_matsurika", "pero"),
    (1525363, 77, 367, "kazamine_matsurika", "pero"),
    (1525363, 368, 384, "tsukitachibana_hime", "usume shirou"),
    (1525363, 387, 431, "tsukitachibana_hime", "usume shirou"),
    (1525363, 438, 675, "tsukitachibana_hime", "usume shirou"),
    (1525363, 676, 893, "shinonome_ruri", "gijang"),
    # Akatsuki Yureru Koi Akari
    (1807851, 24, 131, "kazamine_setsugekka", "pero"),
    (1807851, 132, 216, "suzakuin_momiji", "pero"),
    (1807851, 217, 308, "kuki_asahi", "usume shirou"),

    (1972660, 1, 2000, "suzakuin_momiji", "pero"),
    (1976332, 1, 2000, "kuki_asahi", "usume shirou"),
    # Setsuna ni Kakeru Koi Hanabi
    (2836561, 46, 141, "suzakuin_nadeshiko", "pero"),
    (2836561, 142, 217, "takigawa_komari", "haiba"),
    (2836561, 218, 274, "hanabusa_palvi", "usume shirou"),
    (3042507, 1, 282, "takigawa_komari", "haiba"),
    (3042508, 1, 274, "suzakuin_nadeshiko", "pero"),

    # 9-nine-
    (3655617, 1, 256, "kujou_miyako", "izumi tsubasu"),
    (3655619, 1, 148, "niimi_sora", "izumi tsubasu"),
    (3655622, 1, 441, "kousaka_haruka", "izumi tsubasu"),
    (3655622, 457, 544, "kujou_miyako", "izumi tsubasu"),
    (3655623, 1, 236, "yuuki_noa", "izumi tsubasu"),

    (1056030, 1, 90, "kujou_miyako", "izumi tsubasu"),
    (1215917, 16, 86, "niimi_sora", "izumi tsubasu"),
    (1403990, 13, 115, "kousaka_haruka", "izumi tsubasu"),
    (1620258, 33, 176, "yuuki_noa", "izumi tsubasu"),

    (2392344, 1, 99, "kujou_miyako", "izumi tsubasu"),
    (2392344, 100, 171, "niimi_sora", "izumi tsubasu"),
    (2392344, 174, 246, "kousaka_haruka", "izumi tsubasu"),
    (2392344, 249, 402, "yuuki_noa", "izumi tsubasu"),

    # lose
    (3511654, 1, 2000, None, "cura"),
    (3511487, 1, 2000, None, "cura"),
    # Ano Ko wa Ore kara Hanarenai
    (1011409, 2, 229, "kamishiro_aoi", "niro"),
    (1011409, 241, 388, "miyama_haruka", "niro"),
    (1011409, 389, 578, "naruse_manami_(ano_ko_wa_ore_kara_hanarenai)", "usume shirou"),
    (1011409, 759, 891, "sakurai_yuzuki", "usume shirou"),
    # Harvest OverRay
    (762959, 2, 126, "tamaki_yuuka", "usume shirou"),
    (762959, 127, 231, "mikami_lilia", "niro"),
    (762959, 232, 361, "ouno_sumi", "niro"),
    (762959, 362, 507, "yatsurugi_komachi", "usume shirou"),

    (1134317, 30, 575, "yatsurugi_komachi", "usume shirou"),
    (1134317, 576, 881, "mikami_lilia", "niro"),
    (1134317, 1363, 2000, "ouno_sumi", "niro"),
    (1134330, 1, 32, "ouno_sumi", "niro"),
    (1134330, 41, 332, "tamaki_yuuka", "usume shirou"),
    # Chiccha Love Apart
    (714476, 4, 11, "komori_hinata", "usume shirou"),
    (714476, 12, 56, "kuramoto_mutsumi", "pikazo"),
    (714476, 57, 69, "torikai_nina", "pikazo"),
    (714476, 70, 79, "sakurabashi_takano", "usume shirou"),
    (714476, 80, 259, "hayami_ayaka", "massan"),
    (714476, 260, 559, "komori_hinata", "usume shirou"),
    (714476, 560, 832, "kuramoto_mutsumi", "pikazo"),
    (714476, 833, 1090, "torikai_nina", "pikazo"),
    (714476, 1091, 1370, "sakurabashi_takano", "usume shirou"),

    (1116650, 1, 120, None, "pikazo"),
    (1116650, 121, 856, "hayami_ayaka", "massan"),
    (1116650, 1049, 2000, "komori_hinata", "usume shirou"),

    (1116649, 1, 677, "komori_hinata", "usume shirou"),
    (1116649, 678, 869, None, "usume shirou"),
    (1116649, 870, 2000, "kuramoto_mutsumi", "pikazo"),

    (1116648, 1, 238, "kuramoto_mutsumi", "pikazo"),
    (1116648, 239, 1774, "torikai_nina", "pikazo"),
    (1116648, 1829, 2000, "sakurabashi_takano", "usume shirou"),

    (1116647, 1, 1530, "sakurabashi_takano", "usume shirou"),
    # Lautes Alltags -Herrenlose Katze und Teehaus-
    (782112, 2, 276, "takanashi_nanase", "sorai shinya"),
    (782112, 277, 456, "kasahara_himari", "sorai shinya"),
    (782112, 457, 683, "sakurai_sana", "sorai shinya"),
    (782112, 684, 964, "hirasawa_kaede", "sorai shinya"),
    (782393, 127, 882, "takanashi_nanase", "sorai shinya"),
    (782393, 883, 1701, "kasahara_himari", "sorai shinya"),
    (782393, 1702, 2000, "sakurai_sana", "sorai shinya"),
    (782586, 1, 370, "sakurai_sana", "sorai shinya"),
    (782586, 371, 1168, "hirasawa_kaede", "sorai shinya"),
    # Yomegami - My Sweet Goddess!
    (998648, 12, 101, "narukawa_iris", "hisama_kumako"),
    (998648, 139, 243, "hakari_mari", "mikeou"),
    (998648, 256, 352, "ichijiku_mikoto", "shiraichigo"),
    (998648, 363, 468, "riko", "suimya"),

    (998648, 469, 493, "narukawa_iris", "hisama_kumako"),
    (998648, 495, 512, "hakari_mari", "mikeou"),
    (998648, 513, 533, "ichijiku_mikoto", "shiraichigo"),
    (998648, 534, 559, "riko", "suimya"),

    (3525197, 2, 67, "narukawa_iris", "hisama_kumako"),
    (3525197, 68, 134, "hakari_mari", "mikeou"),
    (3525197, 135, 182, "ichijiku_mikoto", "shiraichigo"),
    (3525197, 183, 271, "riko", "suimya"),

    # Yome no Imouto to H na Kankei ni Natte Yabai!?
    (1678091, 1, 163, "fujimura_hinata", "maccha reika"),
    (1678091, 164, 283, None, "maccha reika"),

    (1669463, 1, 111, "fujimura_hinata", "maccha reika"),
    (1669463, 124, 142, "fujimura_hinata", "maccha reika"),
    (1669463, 152, 168, "fujimura_hinata", "maccha reika"),
    (1669463, 177, 203, "fujimura_hinata", "maccha reika"),

    # Traveling Stars
    (1001634, 1, 147, "zirconia_salvatore", "matsushita makako"),

    (847945, 2, 58, "eliza_roses_crawford", "uonuma yuu"),
    (847945, 59, 205, "finn_sheard", "takayaki"),
    (847945, 206, 381, "zirconia_salvatore", "matsushita makako"),
    (847945, 382, 538, "eliza_roses_crawford", "uonuma yuu"),
    (847945, 539, 774, "chloe_de_prelune", "suzuhira hiro"),
    (847945, 775, 830, "roco_misha", "naenae"),
    (847945, 831, 955, "lea_schelling_von_wolfsburg", "naenae"),
    (847945, 995, 1191, "oumi_saaya", "matsushita makako"),
    # Koi Suru Amairo Homestay -Ryuugakusei wa Wanko-kei Osananajimi
    (2254599, 1, 155, "mary_mea_heart", "hiiragi ringo"),
    # Namaiki Yume-chan wa Onii to Mechakucha H Shitai
    (2520412, 1, 144, "hinamori_yume", "hiiragi ringo"),
    # The Rising Sun Marriage
    (3042610, 1, 1, "alphine_midill", "hiiragi ringo, nanotaro, mizuki yuuma"),
    (3042610, 490, 690, "alphine_midill", "hiiragi ringo"),
    (3042610, 2, 208, "ria", "nanotaro"),
    (3042610, 209, 489, "chloe_rouen", "mizuki yuuma"),
    (3042610, 691, 883, None, " "),
    (3435331, 78, 362, "ria", "nanotaro"),
    (3435331, 363, 562, "chloe_rouen", "mizuki yuuma"),
    (3435331, 563, 737, "alphine_midill", "hiiragi ringo"),
    (3435332, 1, 285, "ria", "nanotaro"),
    (3435332, 286, 485, "chloe_rouen", "mizuki yuuma"),
    (3435332, 486, 660, "alphine_midill", "hiiragi ringo"),
    # Bakappuru Supplement
    (3658306, 4, 641, "kurumi_akiha", "kiba_satoshi"),
    (3658306, 642, 1297, "frederica_ahlqvist", "gintarou"),
    (3658306, 1306, 2000, "ayase_rena", "emily"),
    (3658307, 2, 553, "shishio_rin", "emily"),
    # Study Steady
    (1491074, 915, 1965, "maisaka_mai", "emily"),
    (1491075, 837, 1423, "omaezaki_yuu", "emily"),
    (1919798, 1, 268, "omaezaki_yuu", "emily"),
    (1491074, 1, 2000, None, "kiba_satoshi"),
    (1491075, 1, 2000, None, "kiba_satoshi"),
    # Study § Steady 2
    (2361534, 2, 806, "yaezawa_yae", "emily"),
    (2361534, 827, 1748, "mamanoue_yuno", "emily"),
    (2361497, 1, 2000, None, "kiba_satoshi"),
    (2871902, 1, 693, "mamanoue_yuno", "emily"),
    # Golden Marriage
    (705391, 2, 114, "tange_kasumi", "hayakawa halui"),
    (705391, 115, 126, "ichijouji_toko", "hayakawa halui"),
    (705391, 127, 166, "amaya_rei", "hayakawa halui"),
    (705391, 167, 188, "tange_kasumi", "hayakawa halui"),
    (705391, 213, 341, "amaya_rei", "hayakawa halui"),
    (705391, 342, 464, "shimakage_ruri", "hayakawa halui"),
    (705391, 465, 612, "ichijouji_toko", "hayakawa halui"),
    (705391, 613, 727, "kasugano_yukariko", "hayakawa halui"),
    (705391, 728, 776, "marika_von_wittelsbach", "hayakawa halui"),

    (799691, 9, 40, "tange_kasumi", "hayakawa halui"),
    (799691, 41, 106, "marika_von_wittelsbach", "hayakawa halui"),
    (799691, 107, 169, "amaya_rei", "hayakawa halui"),
    (799691, 170, 206, "shimakage_ruri", "hayakawa halui"),
    (799691, 207, 250, "ichijouji_toko", "hayakawa halui"),
    (799691, 283, 306, "kasugano_yukariko", "hayakawa halui"),

    (799691, 313, 374, "tange_kasumi", "hayakawa halui"),
    (799691, 375, 433, "marika_von_wittelsbach", "hayakawa halui"),
    (799691, 438, 466, "marika_von_wittelsbach", "hayakawa halui"),
    (799691, 467, 508, "amaya_rei", "hayakawa halui"),
    (799691, 509, 577, "shimakage_ruri", "hayakawa halui"),
    (799691, 578, 623, "ichijouji_toko", "hayakawa halui"),
    (799691, 624, 670, "kasugano_yukariko", "hayakawa halui"),
    # HoneDevi! Honey&Devil
    (1068260, 6, 220, "takamiya_ouka", "hayakawa halui"),
    (1068260, 221, 379, "kougousaki_ruri", "hayakawa halui"),
    (1068260, 382, 525, "toudou_aoi_(hanidebi)", "hayakawa halui"),
    (1068260, 526, 697, "nishinozono_kaoruko", "hayakawa halui"),
    (1068260, 771, 777, "nishinozono_kaoruko", "hayakawa halui"),
    # Nakadashi Trilogy
    (1582264, 2, 477, "unazuki_sakuya", "koku"),
    (1582264, 478, 984, "hinohara_haruna", "koku"),
    (1582264, 985, 1056, "unazuki_sakuya, hinohara_haruna", "koku"),
    # Himawari no Kyoukai to Nagai Natsuyasumi
    (1333157, 1, 177, "natsusaki_yomi", "inugami kira"),
    # Sakura no Uta -Sakura no Mori no Ue wo Mau-
    (866422, 1, 145, "misakura_rin", "inugami kira"),
    (866422, 146, 246, "natsume_shizuku", "inugami kira"),
    (866422, 247, 299, "hikawa_rina", "inugami kira"),
    (866422, 301, 412, "hikawa_rina", "inugami kira"),
    (866422, 413, 449, "kawachino_yuumi", "inugami kira"),
    (866422, 450, 632, "toritani_makoto", "kagome"),
    (866422, 633, 751, "natsume_ai", "kagome"),

    (3747631, 2, 34, "natsume_ai", "kagome"),
    (3747631, 118, 195, "misakura_rin", "inugami kira"),
    (3747631, 418, 470, "misakura_rin", "inugami kira"),
    (3747631, 835, 937, "toritani_makoto", "kagome"),
    (3747631, 938, 1001, "misakura_rin", "inugami kira"),
    (3747631, 1021, 1076, "hikawa_rina", "inugami kira"),
    (3747631, 1077, 1158, "natsume_shizuku", "inugami kira"),
    # NEKO-MIMI SWEET HOUSEMATES
    (2191267, 2, 83, "mint_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 101, 130, "mint_(uchi_no_pet_jijou), lily_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 131, 149, "mint_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 150, 160, "cacao_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 166, 204, "lily_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 205, 214, "cacao_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 223, 283, "mint_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 284, 294, "cacao_(uchi_no_pet_jijou)", "yano mitsuki"),
    (2191267, 295, 310, "mint_(uchi_no_pet_jijou)", "yano mitsuki"),
    # Emuria ~Ore ga Do-M ni Natta no wa Dou Kangaete mo Omaera ga Warui~
    (878324, 5, 8, "aso_nozomi", "xe"),
    (878324, 13, 24, "aso_nozomi", "xe"),
    (878324, 71, 211, "tsumadu_fumino", "mitsumomo mam"),
    (878324, 212, 361, "aso_nozomi", "xe"),
    (878324, 362, 498, "tsumadu_yasuna", "mitsumomo mam"),
    (878324, 499, 636, "sagara_haruka", "xe"),
    # Zannen na Oretachi no Seishun Jijou.
    (763722, 2, 130, "sumeragi_rinne", "hatori piyoko"),
    (763722, 131, 289, "aikawa_sakuya", "xe"),
    (763722, 290, 433, "nakaoka_chimachi", "mango pudding"),
    (763722, 434, 490, "shinonome_natsume", "hatori piyoko"),
    (763722, 491, 568, "nakaoka_chimachi", "mango pudding"),
    (763722, 569, 748, "shinonome_natsume", "hatori piyoko"),
    (763722, 799, 831, "shinonome_natsume", "hatori piyoko"),
    (763722, 882, 914, "shinonome_natsume", "hatori piyoko"),
    (763722, 749, 788, "hashimoto_takumi", "mizukoshi mayu"),
    (763722, 832, 871, "hashimoto_takumi", "mizukoshi mayu"),
    (763722, 915, 954, "hashimoto_takumi", "mizukoshi mayu"),
    # Boku no Amayaka Seikatsu -Seishou-chou Kankouka, Mainichi Ecchi na Locodol Katsudou!-
    (1295389, 50, 294, "hasekura_niina", "bekotarou"),
    (1295389, 295, 438, "hasekura_eiru", "bekotarou"),
    (1295389, 439, 615, "mikishima_meika", "komeshiro kasu"),

    (1545452, 267, 680, "hasekura_niina", "bekotarou"),
    (1545452, 681, 836, "hasekura_eiru", "bekotarou"),
    (1545452, 1, 266, "mikishima_meika", "komeshiro kasu"),
    (1545452, 837, 1214, None, "komeshiro kasu"),
    # Ojou-sama to Aware na Koshitsuji
    (1067461, 146, 478, "todoroki_karin", "mitsumomo mam"),
    (1067461, 479, 816, "todoroki_minamo", "hatori piyoko"),
    (1067461, 817, 1151, "todoroki_tsukushi", "mitsumomo mam"),
    (1067461, 1152, 1296, "otonashi_mitsuko", "hatori piyoko"),

    (1425336, 2, 273, "todoroki_karin", "mitsumomo mam"),
    (1425336, 274, 885, "todoroki_minamo", "hatori piyoko"),
    (1425336, 886, 1191, "todoroki_tsukushi", "mitsumomo mam"),
    (1425336, 1192, 1341, "otonashi_mitsuko", "hatori piyoko"),
    # Boku no xx wa Ryousei-tachi no Tokken desu!
    (929519, 2, 99, "asakura_yuzuki", "hisama kumako"),
    (929519, 100, 176, "murakami_rino", "sakai minato"),
    (929519, 177, 270, "saitou_kanna", "miyasaka naco"),
    (929519, 271, 349, "olivia_campbell", None),
    # HajiLove
    (1990136, 3, 332, "shinohara_kouta", "k-ko"),
    (1990136, 333, 620, "sonoike_sakurako", "k-ko"),
    (1990136, 621, 924, "yofune_hatsuho", "mango pudding"),
    (1990136, 925, 1215, "hakari_yui", "mango pudding"),

    (2188278, 5, 148, "sonoike_sakurako", "k-ko"),
    (2188278, 149, 276, "hakari_yui", "mango pudding"),
    (2188278, 277, 308, "sonoike_sakurako", "k-ko"),
    (2188278, 309, 317, "hakari_yui", "mango pudding"),

    (2232039, 2, 149, "shinohara_kouta", "k-ko"),
    (2232039, 150, 362, "yofune_hatsuho", "mango pudding"),
    (2232039, 363, 510, "shinohara_kouta", "k-ko"),
    (2232039, 511, 721, "yofune_hatsuho", "mango pudding"),

    # hamidashi creative
    (1740846, 5, 13, "izumi_hiyori", "utsunomiya tsumire"),
    (1740846, 110, 266, "nishiki_asumi", "utsunomiya tsumire"),
    (1740846, 267, 423, "izumi_hiyori", "utsunomiya tsumire"),
    (1740846, 424, 559, "tokiwa_kano", "utsunomiya tsumire"),
    (1740846, 560, 677, "kamakura_shio", "utsunomiya tsumire"),

    (1741935, 1, 170, "ryuukan_ameri", "utsunomiya tsumire"),
    (1741935, 171, 751, "nishiki_asumi", "utsunomiya tsumire"),
    (1741935, 809, 1411, "izumi_hiyori", "utsunomiya tsumire"),
    (1741935, 1412, 2000, "tokiwa_kano", "utsunomiya tsumire"),

    (1741936, 1, 80, "tokiwa_kano", "utsunomiya tsumire"),
    (1741936, 81, 278, "izumi_miri", "utsunomiya tsumire"),
    (1741936, 279, 309, "hijiri_ririko", "utsunomiya tsumire"),
    (1741936, 310, 687, "kamakura_shio", "utsunomiya tsumire"),

    (1741935, 1, 2000, None, "utsunomiya tsumire"),
    (1741936, 1, 2000, None, "utsunomiya tsumire"),

    (2384486, 5, 332, "ryuukan_ameri", "utsunomiya tsumire"),
    (2384486, 333, 512, "nishiki_asumi", "utsunomiya tsumire"),
    (2384486, 527, 745, "izumi_hiyori", "utsunomiya tsumire"),
    (2384486, 746, 896, "tokiwa_kano", "utsunomiya tsumire"),
    (2384486, 897, 1061, "kamakura_shio", "utsunomiya tsumire"),


    (2390603, 1, 157, "nishiki_asumi", "utsunomiya tsumire"),
    (2390603, 263, 419, "izumi_hiyori", "utsunomiya tsumire"),
    (2390603, 420, 555, "tokiwa_kano", "utsunomiya tsumire"),
    (2390603, 556, 673, "kamakura_shio", "utsunomiya tsumire"),
    (2390603, 674, 759, "nishiki_asumi", "utsunomiya tsumire"),
    (2390603, 814, 916, "izumi_hiyori", "utsunomiya tsumire"),
    (2390603, 917, 982, "tokiwa_kano", "utsunomiya tsumire"),
    (2390603, 983, 1043, "kamakura_shio", "utsunomiya tsumire"),

    (3328453, 10, 356, "hijiri_ririko", "utsunomiya tsumire"),
    (3328374, 8, 354, "hijiri_ririko", "utsunomiya tsumire"),
    (3328374, 3, 2000, None, "utsunomiya tsumire"),
    # Love Commu
    (1389165, 2, 287, "saionji_shouko", "naenae"),
    (1389165, 288, 609, "tsukimiya_rin", "naenae"),
    (1389165, 614, 1081, "takatsukasa_makoto", "kiduki erika"),
    (1389165, 1062, 1174, "akera_kurumi", "naenae"),
    (1389165, 1175, 1390, "ikoma_mitsuru", "kiduki erika"),
    (1389165, 1391, 1431, "yakumo_naru", "kiduki erika"),

    (1404973, 1, 420, "saionji_shouko", "naenae"),
    (1404973, 421, 820, "tsukimiya_rin", "naenae"),
    (1404973, 821, 1460, "takatsukasa_makoto", "kiduki erika"),
    (1404973, 1461, 1704, "akera_kurumi", "naenae"),
    (1404973, 1705, 1944, "ikoma_mitsuru", "kiduki erika"),
    (1404973, 1945, 2000, "yakumo_naru", "kiduki erika"),
    (1404977, 1, 184, "yakumo_naru", "kiduki erika"),
    (1404977, 185, 2000, None, "naenae"),
    # Kimi to Boku to no Kishi no Hibi -Rakuen no Chevalier-
    (992571, 44, 141, "saionji_kei", "ozawa akifumi"),
    (992571, 340, 358, None, "pero"),
    # Hoshifuru Yoru no Farnese
    (1122361, 2, 70, "farnese_atlas", "yukie"),
    (1122361, 71, 123, "jacqueline_sprenger", "yukie"),
    (1122361, 124, 167, "yves_klein", "yukie"),
    (1122361, 168, 213, "orihime_n_astil", "yukie"),
    (1122361, 1, 228, None, "yukie"),
    (1122361, 229, 430, None, "naenae"),
    # Amayakase Kanojo no Iru Seikatsu
    (1506539, 1, 2000, "anya_(amayakase_kanojo_no_iru_seikatsu)", "naenae"),
    (1503641, 1, 2000, "anya_(amayakase_kanojo_no_iru_seikatsu)", "naenae"),
    # Yadourishi Otome Na Chikai To Maho
    (3454489, 2, 182, "mikuri_chami", "mizuki yuuma"),
    (3454489, 271, 275, "kokutan_mare", "hiiragi ringo"),
    (3454489, 319, 496, "torineko_maho", "mizuki yuuma"),
    (3454489, 497, 751, "kokutan_mare", "hiiragi ringo"),
    (3454489, 752, 877, "kashihara_yuyu", "mutou kurihito"),

    (3455080, 154, 243, "kashihara_yuyu", "mutou kurihito"),
    (3455080, 244, 360, "torineko_maho", "mizuki yuuma"),
    (3455080, 361, 486, "mikuri_chami", "mizuki yuuma"),
    (3455080, 487, 585, "kokutan_mare", "hiiragi ringo"),
    (3455080, 586, 603, None, "hiiragi ringo"),
    (3455080, 826, 915, "kashihara_yuyu", "mutou kurihito"),
    (3455080, 916, 1032, "torineko_maho", "mizuki yuuma"),
    (3455080, 1033, 1158, "mikuri_chami", "mizuki yuuma"),
    (3455080, 1159, 1257, "kokutan_mare", "hiiragi ringo"),
    (3455080, 1258, 1275, None, "hiiragi ringo"),
    # iegami_nyoubou
    (1322348, 1, 2000, "nanashi_nekomata", None),
    (1381507, 1, 672, "nanashi_nekomata", None),
    (1809019, 738, 1123, "nanashi_nekomata", None),
    # magicalic
    (1087664, 2, 163, "kawasumi_yurika", "mikagami mamizu"),
    (1087664, 164, 289, "fana_arsim", "mikagami mamizu"),
    (1087664, 290, 458, "charlles_faltesia", "mikagami mamizu"),
    (1087664, 459, 588, "emilia_purihu_takamine", "mikagami mamizu"),
    (1087664, 589, 623, "saraira", "mikagami mamizu"),
    (3450022, 1, 1188, "kawasumi_yurika", "mikagami mamizu"),
    (3450022, 1189, 1860, "fana_arsim", "mikagami mamizu"),
    (3450023, 1, 1026, "emilia_purihu_takamine", "mikagami mamizu"),
    (3450024, 1, 1080, "charlles_faltesia", "mikagami mamizu"),
    (3450024, 1081, 1326, "saraira", "mikagami mamizu"),
    # Osananajimi Ojou-sama to H de Himitsu na Dousei Seikatsu
    (2622654, 1, 1548, "minegishi_yuuma", "mitsuhamochi."),
    # Jewelry Hearts Academia
    (2284346, 34, 79, "arianna_heartbell", "shiratama"),
    (2284346, 80, 123, "berka_triad", "shiratama"),
    (2284346, 124, 177, "mare_ashley-pecker", "shiratama"),
    (2284346, 178, 224, "ruby_(jewelry_hearts_academia)", "shiratama"),
    (2284346, 532, 576, "arianna_heartbell", "shiratama"),
    (2284346, 577, 623, "berka_triad", "shiratama"),
    (2284346, 624, 664, "mare_ashley-pecker", "shiratama"),
    (2284346, 665, 714, "ruby_(jewelry_hearts_academia)", "shiratama"),

    (2397624, 1, 282, "arianna_heartbell", "shiratama"),
    (2397624, 283, 690, "berka_triad", "shiratama"),
    (3115867, 1, 2000, "mare_ashley-pecker", "shiratama"),
    (2397624, 1204, 1342, "ruby_(jewelry_hearts_academia)", "shiratama"),

    (3255696, 14, 21, "arianna_heartbell", "shiratama"),
    (3255696, 22, 31, "berka_triad", "shiratama"),
    (3255696, 32, 43, "mare_ashley-pecker", "shiratama"),
    (3255696, 44, 51, "ruby_(jewelry_hearts_academia)", "shiratama"),
    (3255696, 219, 240, "arianna_heartbell", "shiratama"),
    (3255696, 241, 269, "berka_triad", "shiratama"),
    (3255696, 270, 293, "mare_ashley-pecker", "shiratama"),
    (3255696, 294, 316, "ruby_(jewelry_hearts_academia)", "shiratama"),
    (3255696, 172, 185, "milia_ehlendorf", "shiratama"),
    (3255696, 376, 417, "milia_ehlendorf", "shiratama"),

    # Secret Agent
    (1648992, 38, 133, "shirogane_kagura", "odawara hakone"),
    (1648992, 134, 232, "kanon_mayfield", "mutou kurihito"),
    (1648992, 241, 348, "renjouji_mai", "ayase hazuki"),
    (1648992, 354, 466, "amenomori_yui", "awayume"),

    (1657228, 769, 2000, "shirogane_kagura", "odawara hakone"),
    (1657228, 1, 768, "kanon_mayfield", "mutou kurihito"),
    (1657204, 1, 614, "renjouji_mai", "ayase hazuki"),
    (1657204, 615, 1094, "amenomori_yui", "awayume"),

    (1902036, 138, 205, "shirogane_kagura", "odawara hakone"),
    (1902036, 206, 260, "kanon_mayfield", "mutou kurihito"),
    (1902036, 261, 370, "renjouji_mai", "ayase hazuki"),
    (1902036, 371, 437, "amenomori_yui", "awayume"),

    (1904262, 6, 38, "shirogane_kagura", "odawara hakone"),
    (1904262, 39, 65, "kanon_mayfield", "mutou kurihito"),
    (1904262, 66, 98, "renjouji_mai", "ayase hazuki"),
    (1904262, 99, 120, "shirogane_kagura", "odawara hakone"),
    (1904262, 121, 140, "kanon_mayfield", "mutou kurihito"),
    (1904262, 141, 160, "renjouji_mai", "ayase hazuki"),

    (2691844, 36, 131, "shirogane_kagura", "odawara hakone"),
    (2691844, 132, 230, "kanon_mayfield", "mutou kurihito"),
    (2691844, 239, 346, "renjouji_mai", "ayase hazuki"),
    (2691844, 352, 464, "amenomori_yui", "awayume"),

    (2691844, 495, 582, "shirogane_kagura", "odawara hakone"),
    (2691844, 583, 647, "kanon_mayfield", "mutou kurihito"),
    (2691844, 656, 744, "renjouji_mai", "ayase hazuki"),
    (2691844, 747, 826, "amenomori_yui", "awayume"),

    (1902387, 162, 262, "shirogane_kagura", "odawara hakone"),
    (1902387, 1, 161, "kanon_mayfield", "mutou kurihito"),
    (1902387, 263, 698, "renjouji_mai", "ayase hazuki"),
    (1902387, 699, 818, "amenomori_yui", "awayume"),
    # Idol Wars Z
    (1066329, 1, 94, "amane_ai", None),
    (1624559, 112, 121, "amane_ai", None),
    (1624559, 1172, 1181, "amane_ai", None),
    (1624559, 1434, 1443, "amane_ai", None),

    (1624589, 189, 198, "amane_ai", None),
    (1624589, 431, 440, "amane_ai", None),
    (1624589, 955, 964, "amane_ai", None),

    (1624562, 54, 72, "amane_ai", None),
    (1624562, 317, 326, "amane_ai", None),
    (1624562, 1213, 1222, "amane_ai", None),

    (1676574, 1, 11, "amane_ai", None),
    (1676574, 57, 66, "amane_ai", None),

    (1730956, 181, 200, "amane_ai", None),
    (1730956, 301, 310, "amane_ai", None),
    (1730956, 331, 350, "amane_ai", None),

    (3088879, 1, 9, "amane_ai", None),
    (3088879, 340, 349, "amane_ai", None),
    (3088879, 370, 389, "amane_ai", None),
    (3088879, 910, 919, "amane_ai", None),
    (3088879, 1020, 1029, "amane_ai", None),
    (3088879, 1090, 1109, "amane_ai", None),
    (3088879, 1205, 1214, "amane_ai", None),
    (3088879, 1280, 1289, "amane_ai", None),
    (3088879, 1575, 1594, "amane_ai", None),
    (3088879, 1625, 1634, "amane_ai", None),

    (3088880, 1, 20, "amane_ai", None),
    (3088880, 106, 110, "amane_ai", None),

    (3088881, 1, 19, "amane_ai", None),
    (3088881, 49, 67, "amane_ai", None),
    (3088881, 202, 220, "amane_ai", None),
    (3088881, 231, 249, "amane_ai", None),
    (3088881, 460, 464, "amane_ai", None),
    # Twinkle Star Knights
    (3163484, 9, 12, "venus_(twinkle_star_knights)", "kannagi rei"),
    (3163484, 243, 244, "venus_(twinkle_star_knights)", "kannagi rei"),
    (3692789, 14, 19, "venus_(twinkle_star_knights)", "kannagi rei"),
    # Aria the Godslayer
    (2072970, 1, 3, "kuonji_maki", None),
    (2072970, 184, 189, None, "motoi ayumu"),
    (2044170, 3, 4, "kuonji_maki", None),
    # Plus Links
    (2105816, 2, 21, "misaki_himawari", None),
    (2105816, 95, 120, "kanagumo_miyabi", None),
    (2105816, 162, 182, "shiromiya_rin", None),
    (2105816, 183, 203, "kamiyugi_tama", None),
    (2105816, 237, 242, "misaki_himawari", None),
    (2105816, 217, 220, "kanagumo_miyabi", None),
    (2105816, 275, 282, "kamiyugi_tama", None),
    (2105816, 322, 323, "kamiyugi_tama", None),

    (1964286, 419, 421, "misaki_himawari", None),
    (1964286, 434, 438, "kanagumo_miyabi", None),
    (1964286, 445, 447, "shiromiya_rin", None),
    (1964286, 448, 450, "kamiyugi_tama", None),
    # Rensou Relation
    (819614, 5, 72, "sengoku_ichika", "moekibara fumitake"),
    (819614, 147, 206, None, "moekibara fumitake"),
    (819614, 295, 319, None, "moekibara fumitake"),
    (819614, 320, 343, "sengoku_ichika", "moekibara fumitake"),
    (819614, 370, 384, None, "moekibara fumitake"),
    (819614, 1, 2000, None, "nanaroba hana"),
    (1124566, 67, 126, None, "moekibara fumitake"),
    (1124566, 372, 505, "sengoku_ichika", "moekibara fumitake"),
    (1124566, 1, 2000, None, "nanaroba hana"),
    # Kodomo no Asobi
    (877935, 4, 87, "sera_(kodomo_no_asobi)", "moekibara fumitake"),
    (877935, 88, 141, "tsurumaki_yuzuriha", "annie"),
    (877935, 142, 207, "hinako_michiru", "nanaroba hana"),
    (877935, 208, 246, "niu_katsumi_(kodomo_no_asobi)", "hatori piyoko"),
    (877935, 247, 331, "sera_(kodomo_no_asobi)", "moekibara fumitake"),
    (877935, 332, 435, "tsurumaki_yuzuriha", "annie"),
    (877935, 436, 566, "hinako_michiru", "nanaroba hana"),
    (877935, 567, 646, "niu_katsumi_(kodomo_no_asobi)", "hatori piyoko"),

    (1124523, 45, 146, "hinako_michiru", "nanaroba hana"),
    (1124523, 370, 590, "sera_(kodomo_no_asobi)", "moekibara fumitake"),
    (1124523, 709, 842, "tsurumaki_yuzuriha", "annie"),
    (1124523, 843, 882, "hinako_michiru", "nanaroba hana"),
    (1124523, 883, 1091, "niu_katsumi_(kodomo_no_asobi)", "hatori piyoko"),
    # Sweet Homemade
    (2766424, 4, 7, "ezaki_iroha", None),
    (2766424, 32, 66, "ezaki_iroha", None),
    (2766424, 12, 16, "koike_kanon", None),
    (2766424, 98, 128, "koike_kanon", None),
    (2766424, 28, 31, "yamazaki_himariko", None),
    (2766424, 219, 250, "yamazaki_himariko", None),

    (2949996, 6, 10, "ezaki_iroha", None),
    (2949996, 41, 115, "ezaki_iroha", None),
    (2949996, 16, 20, "ezaki_iroha", None),
    (2949996, 12, 16, "koike_kanon", None),
    (2949996, 191, 265, "koike_kanon", None),
    (2949996, 651, 655, "koike_kanon", None),
    (2949996, 36, 40, "yamazaki_himariko", None),
    (2949996, 491, 565, "yamazaki_himariko", None),
    (2949996, 671, 675, "yamazaki_himariko", None),
    # Omokage
    (917558, 3, 170, "ichinose_minato", "amakusa tobari"),
    (917558, 171, 340, "tachibana_gekka", "amakusa tobari"),
    (917558, 341, 493, "suzu_hinami", "amakusa tobari"),

    (918417, 95, 104, "ichinose_minato", "amakusa tobari"),
    (918417, 105, 118, "tachibana_gekka", "amakusa tobari"),
    # Unmei Senjou no Phi
    (753603, 1, 40, None, "moekibara fumitake"),
    (753603, 41, 81, "kuon_nagisa", "nanaroba hana"),
    (753603, 120, 168, None, "moekibara fumitake"),
    (753603, 200, 283, None, "moekibara fumitake"),
    (753603, 284, 359, "kuon_nagisa", "nanaroba hana"),
    (753603, 448, 515, None, "moekibara fumitake"),
    # mono no aware wa sai no koro
    (3534610, 35, 69, "nonomiya_kyouka", "nanaroba hana"),
    (3534610, 70, 120, "kinami_misaki", "nanaroba hana"),
    (3534610, 121, 156, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (3534610, 157, 191, "claire_courtney_claire", "nanaroba hana"),
    (3534610, 268, 286, "nonomiya_kyouka", "nanaroba hana"),
    (3534610, 287, 303, "kinami_misaki", "nanaroba hana"),
    (3534610, 304, 320, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (3534610, 321, 333, "claire_courtney_claire", "nanaroba hana"),
    (3534610, 334, 368, "nonomiya_kyouka", "nanaroba hana"),
    (3534610, 369, 402, "kinami_misaki", "nanaroba hana"),
    (3534610, 403, 436, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (3534610, 437, 469, "claire_courtney_claire", "nanaroba hana"),

    (1122195, 36, 70, "nonomiya_kyouka", "nanaroba hana"),
    (1122195, 71, 121, "kinami_misaki", "nanaroba hana"),
    (1122195, 121, 157, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (1122195, 158, 192, "claire_courtney_claire", "nanaroba hana"),
    (1122195, 278, 287, "nonomiya_kyouka", "nanaroba hana"),
    (1122195, 288, 304, "kinami_misaki", "nanaroba hana"),
    (1122195, 305, 321, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (1122195, 322, 334, "claire_courtney_claire", "nanaroba hana"),
    (1122195, 335, 369, "nonomiya_kyouka", "nanaroba hana"),
    (1122195, 370, 403, "kinami_misaki", "nanaroba hana"),
    (1122195, 404, 437, "kohaku_(mono_no_aware_wa_sai_no_koro.)", "nanaroba hana"),
    (1122195, 438, 470, "claire_courtney_claire", "nanaroba hana"),
    # Koisuru Kokoro to Mahou no Kotoba
    (1230458, 43, 208, "haruharu", "shiromochi sakura"),
    (1230458, 209, 367, "hitohira_kazane", "shiromochi sakura"),
    (1230458, 368, 540, "futaba_mikana", "shiromochi sakura"),
    (1230458, 541, 718, "hoshiyomi_mashiro", "shiromochi sakura"),

    (1434182, 1, 2000, "haruharu", "shiromochi sakura"),
    (1434567, 1, 2000, "hitohira_kazane", "shiromochi sakura"),
    (1434574, 1, 2000, "futaba_mikana", "shiromochi sakura"),
    (1434594, 1, 2000, "hoshiyomi_mashiro", "shiromochi sakura"),
    (1434600, 1, 1261, "hoshiyomi_mashiro", "shiromochi sakura"),
    # Panical Confusion
    (1576268, 11, 131, "hanabishi_honoka", "shiromochi sakura"),
    (800033, 12, 62, "hanabishi_honoka", "shiromochi sakura"),
    # PURELY x CATION
    (929041, 7, 395, "minami_mai", "nanaroba hana"),
    (929041, 396, 714, "aoi_sumire", "nanaroba hana"),
    (929041, 715, 989, "natsuki_hikari", "nanaroba hana"),
    (929041, 990, 1211, "ayase_touka", "nanaroba hana"),

    (1413861, 5, 116, "minami_mai", "nanaroba hana"),
    (1413861, 117, 235, "aoi_sumire", "nanaroba hana"),
    (1413861, 236, 316, "natsuki_hikari", "nanaroba hana"),
    (1413861, 317, 439, "ayase_touka", "nanaroba hana"),

    (1413861, 449, 451, "minami_mai", "nanaroba hana"),
    (1413861, 440, 443, "aoi_sumire", "nanaroba hana"),
    (1413861, 452, 454, "natsuki_hikari", "nanaroba hana"),
    (1413861, 444, 448, "ayase_touka", "nanaroba hana"),
    # PRETTY x CATION 2
    (1218763, 10, 268, "himekawa_honami", "asami asami"),
    (1218763, 269, 522, "ashiya_suzuka", "asami asami"),
    (1218763, 523, 790, "kurashiki_azusa", "asami asami"),
    (1218763, 791, 1039, "hayase_chitose", "asami asami"),
    # Hakuchuumu no Aojashin
    (1740823, 10, 447, "yonagi_(hakuchuumu_no_aojashin)", "shimofuri"),
    (1740823, 461, 720, "hatano_rin", "gyokuto_b"),
    (1740823, 798, 1030, "olivia_berry", "gyokuto_b"),
    (1740823, 1176, 1439, "momonouchi_sumomo", "shimofuri"),

    (1741452, 692, 859, "yonagi_(hakuchuumu_no_aojashin)", "shimofuri"),
    (1741452, 860, 1040, "hatano_rin", "gyokuto_b"),
    (1741452, 1041, 1236, "olivia_berry", "gyokuto_b"),
    (1741452, 1237, 1455, "momonouchi_sumomo", "shimofuri"),

    (1743085, 1, 520, "yonagi_(hakuchuumu_no_aojashin)", "shimofuri"),
    (1743085, 648, 1202, "hatano_rin", "gyokuto_b"),
    (1743085, 1203, 2000, "olivia_berry", "gyokuto_b"),
    (1743088, 1, 302, "momonouchi_sumomo", "shimofuri"),
    # Kimi to Yumemishi
    (960882, 3, 108, "hiiragi_marina", None),
    # Newton to Ringo no Ki
    (1067353, 2, 230, "alice_bedford", "bekotarou"),
    (1067353, 231, 577, "utakane_yotsuko", "bekotarou"),
    (1067353, 578, 765, "tsukumo_haru_(newton_to_ringo_no_ki)", "bekotarou"),
    (1067353, 1006, 1068, "emmy_felton", "shimofuri"),

    (1276802, 2, 234, "alice_bedford", "bekotarou"),
    (1276802, 235, 479, "utakane_yotsuko", "bekotarou"),
    (1276802, 480, 654, "tsukumo_haru_(newton_to_ringo_no_ki)", "bekotarou"),
    (1276802, 851, 909, "emmy_felton", "shimofuri"),
    # AstralAir no Shiroki Towa
    # Yubisaki Connection
    (1901135, 7, 342, "tachibana_iori", "ayase hazuki"),
    (1901135, 343, 589, "akizuki_mikoto", "ayase hazuki"),
    (1901135, 590, 908, "futaba_natsuho", "ayase hazuki"),
    (1901135, 909, 1164, "sakurazaka_yuzuki", "ayase hazuki"),

    (2152355, 4, 252, "akizuki_mikoto", "ayase hazuki"),
    (2152355, 253, 483, "sakurazaka_yuzuki", "ayase hazuki"),

    (2206764, 2, 247, "tachibana_iori", "ayase hazuki"),
    (2206764, 248, 507, "futaba_natsuho", "ayase hazuki"),
    (2206764, 508, 641, "tachibana_iori", "ayase hazuki"),
    (2206764, 642, 774, "futaba_natsuho", "ayase hazuki"),
    # Templa!!
    (1215962, 211, 211, "sakurai_shio", "takano yuki"),
    (1215962, 24, 137, "kosaka_miyori", "mitsumomo mam"),
    (1215962, 138, 286, "fujimiya_rinne", "tanihara natsuki"),
    (1215962, 287, 419, "koiwai_sena", "kisaragi yuu"),
    (1215962, 420, 532, "sakurai_shio", "takano yuki"),
    # Tarareba
    (939660, 2, 172, "miyamae_heinrich_komachi", "inuzumi masaki"),
    (939660, 173, 310, "michimura_yuzuko", "takano yuki"),
    (939660, 311, 451, "ooyama_riri", "inuzumi masaki"),
    (939660, 452, 561, "thea_(tarareba)", "inuzumi masaki"),
    # Ore no Kanojo no Uraomote
    (1331513, 2, 101, "sakuragi_yui", "ariko youichi"),
    (1331513, 102, 186, "tomiya_nazuki", "inuzumi masaki"),
    (1331513, 188, 294, "uesugi_akeno", "inuzumi masaki"),
    (1331513, 295, 372, "narugasaki_kanna", "tateha"),

    (1105223, 1, 244, "sakuragi_yui", "ariko youichi"),
    (1105223, 245, 425, "tomiya_nazuki", "inuzumi masaki"),
    (1105223, 426, 647, "uesugi_akeno", "inuzumi masaki"),
    (1105223, 648, 796, "narugasaki_kanna", "tateha"),
    # RE:D Cherish
    (2745978, 1, 216, "rouge_wentworth", "shuto haruka"),
    (2745978, 348, 623, "rouge_wentworth", "shuto haruka"),
    (2745978, 624, 625, "desperado", "saikirider"),
    (2745980, 1, 207, "desperado", "saikirider"),

    (2447977, 40, 267, "unica_rasperanza", "pero"),
    (2447977, 268, 281, "rouge_wentworth", "shuto haruka"),
    (2447977, 282, 292, "desperado", "saikirider"),

    (2152721, 57, 189, "unica_rasperanza", "pero"),
    (2152721, 190, 339, "rouge_wentworth", "shuto haruka"),
    (2152721, 340, 442, "desperado", "saikirider"),

    (2154979, 60, 192, "unica_rasperanza", "pero"),
    (2154979, 193, 343, "rouge_wentworth", "shuto haruka"),
    (2154979, 344, 446, "desperado", "saikirider"),

    (2447968, 1, 249, "unica_rasperanza", "pero"),
    (2447968, 250, 350, "rouge_wentworth", "shuto haruka"),
    (2447968, 351, 480, "desperado", "saikirider"),
    (2447968, 911, 1070, "unica_rasperanza", "pero"),
    (2447968, 1071, 1091, "rouge_wentworth", "shuto haruka"),
    (2447968, 1092, 1173, "desperado", "saikirider"),
    # Neko-nin
    (3659750, 1, 2000, None, "takano yuki"),
    (3688156, 1, 2000, None, "takano yuki"),
    # Koi ni, Kanmi o Soete
    (1165254, 52, 116, "shionomiya_richer", "miyasaka miyu"),
    (1165254, 117, 205, "niwasaka_rira", "miyasaka miyu"),

    (1348014, 9, 93, "hoshigaoka_ciel", "miyasaka miyu"),
    (1348014, 94, 154, "inae_koron", "miyasaka naco"),
    # Hokenshitsu no Sensei to Koakuma na Kaichou
    (2230669, 1, 134, "tsukimori_rin", "santa matsuri"),
    (1785863, 12, 144, "shirobana", "santa matsuri"),
    (2175994, 1, 73, "otohime_(santa_matsuri)", "santa matsuri"),
    (2175994, 81, 129, "otohime_(santa_matsuri)", "santa matsuri"),
    # Animal☆Panic
    (1536541, 5, 34, "shirafuji_yuna", "miyasaka miyu"),
    (1536541, 35, 64, "morino_hinata", "miyasaka miyu"),
    (1536541, 65, 94, "kisaki_shiori", "miyasaka naco"),
    (1536541, 95, 118, "minazuki_rune", "miyasaka miyu"),
    (1536541, 119, 192, "shirafuji_yuna", "miyasaka miyu"),
    (1536541, 193, 252, "morino_hinata", "miyasaka miyu"),
    (1536541, 253, 311, "kisaki_shiori", "miyasaka naco"),
    (1536541, 312, 378, "minazuki_rune", "miyasaka miyu"),
    # Kimi no Hitomi ni Hit Me
    (1021932, 3, 108, "hinata_hitomi", "nekonyan"),
    (1021932, 109, 199, "tsukahata_miko", "kurasawa moko"),
    (1021932, 209, 323, "hataya_shiina", "hisama kumako"),
    (1021932, 324, 440, "kurose_tsubasa", "nekonyan"),
    # Natsuiro Ramune
    (1206260, 5, 36, "sakashita_yayoi", "sesena yau"),
    (1206260, 37, 60, "kojima_misaki", "ichiri"),
    (1206260, 61, 94, "yamazaki_kanako", "sesena yau"),
    (1206260, 95, 126, "honjou_yuuki", "ichiri"),
    (1206260, 138, 228, "sakashita_yayoi", "sesena yau"),
    (1206260, 229, 312, "kojima_misaki", "ichiri"),
    (1206260, 313, 386, "yamazaki_kanako", "sesena yau"),
    (1206260, 387, 471, "honjou_yuuki", "ichiri"),

    (2341963, 2, 123, "sakashita_yayoi", "sesena yau"),
    (2341963, 124, 230, "kojima_misaki", "ichiri"),
    (2341963, 231, 337, "yamazaki_kanako", "sesena yau"),
    (2341963, 338, 453, "honjou_yuuki", "ichiri"),

    # Haruka na Sora
    (3593963, 3, 53, "kasugano_sora ", "hashimoto takashi"),
    # Koi to Koi Suru Utopia
    (948417, 47, 103, "nishihara_kaho", "kani biimu"),
    (948417, 104, 154, "maibara_yukina", "kani biimu"),
    (948417, 155, 202, "takasugi_nanao", "kani biimu"),
    (948417, 12, 12, "yashiki_moegi", "kani biimu"),
    (948417, 41, 42, "yashiki_moegi", "kani biimu"),
    (948417, 203, 218, "yashiki_moegi", "kani biimu"),
    (948417, 10, 10, "morinaga_rika", "kani biimu"),
    (948417, 45, 46, "morinaga_rika", "kani biimu"),
    (948417, 219, 231, "morinaga_rika", "kani biimu"),
    # Koishiki Manual
    (679855, 92, 227, "ichinose_himeno", "saeki nao"),
    (679855, 228, 354, "hikage_honoka", "saeki nao"),
    (679855, 355, 458, "suzuka_aki", "saeki nao"),
    (679855, 459, 559, "haneshiro_amane", "saeki nao"),
    # Angel Ring
    (254267, 2, 123, "mika_alsted_heine", "yamakaze ran"),
    (254267, 124, 237, "fujii_sumika", "yamakaze ran"),
    (254267, 238, 363, "toomi_sana", "yamakaze ran"),
    (254267, 364, 465, "suou_mitsuru", "saeki nao"),
    (254267, 466, 516, "rukia_luminous_suiren", "saeki nao"),
    (254267, 517, 572, "shiki_azusa", "saeki nao"),

    (1087377, 2, 123, "mika_alsted_heine", "yamakaze ran"),
    (1087377, 124, 237, "fujii_sumika", "yamakaze ran"),
    (1087377, 238, 363, "toomi_sana", "yamakaze ran"),
    (1087377, 364, 465, "suou_mitsuru", "saeki nao"),
    (1087377, 466, 516, "rukia_luminous_suiren", "saeki nao"),
    (1087377, 517, 572, "shiki_azusa", "saeki nao"),
    # Tsugihagi Make Peace
    (1133867, 3, 137, "ayasegawa_ayase", "kaniya shiku"),
    (1133867, 138, 283, "toono_nozomi", "tomo"),
    (1133867, 284, 414, "chayahara_hatsune", "mizukoshi mayu"),
    (1133867, 415, 549, "hanakouji_iroha ", "tomo"),
    # Select Oblige
    (2999345, 157, 504, "tateshina_eve", "yuzuna hiyo"),
    (2999345, 505, 833, "isshiki_kaname", "yuzuna hiyo"),
    (2999345, 834, 1198, "yato_kukuru", "yuzuna hiyo"),
    (2999345, 1199, 1530, "touri", "yuzuna hiyo"),

    (3000207, 1, 580, "isshiki_kaname", "yuzuna hiyo"),
    (3000207, 581, 2000, "tateshina_eve", "yuzuna hiyo"),
    (3000208, 1, 1120, "yato_kukuru", "yuzuna hiyo"),
    (3000209, 1, 936, "touri", "yuzuna hiyo"),
    (3000440, 65, 944, "tateshina_eve", "yuzuna hiyo"),
    (3000440, 1111, 1550, "isshiki_kaname", "yuzuna hiyo"),
    (3000441, 1, 616, "yato_kukuru", "yuzuna hiyo"),
    (3000441, 1044, 2000, "touri", "yuzuna hiyo"),
    (3000207, 1, 2000, None, "yuzuna hiyo"),
    (3000208, 1, 2000, None, "yuzuna hiyo"),
    (3000209, 1, 2000, None, "yuzuna hiyo"),
    (3000440, 1, 2000, None, "yuzuna hiyo"),
    (3000441, 1, 2000, None, "yuzuna hiyo"),
    # Bosei Kanojo
    (1204236, 1, 2000, "tachibana_miori", "oryou"),
    (1376030, 1, 1088, "tachibana_miori", "oryou"),

    (1403919, 1, 2000, "sakurai_iyo", "oryou"),
    (1404447, 1, 956, "sakurai_iyo", "oryou"),
    # Aibeya
    (1370401, 1, 2000, "hayami_aki", "oryou"),
    (3443132, 1, 2000, "hayami_aki", "oryou"),

    (1807912, 1, 2000, "misora_saku", "oryou"),
    (3443133, 1, 2000, "misora_saku", "oryou"),
    # Re CATION ~Melty Healing~
    (1690402, 6, 227, "tsukinose_riho", "oryou"),
    (1690402, 228, 460, "futagawa_haru", "oryou"),
    (1690402, 461, 697, "tsubaki_hinako", "oryou"),

    (1946393, 1, 20, "tsukinose_riho", "oryou"),
    (1946393, 21, 38, "futagawa_haru", "oryou"),
    (1946393, 39, 56, "tsubaki_hinako", "oryou"),

    (1818431, 1, 2000, "tsukinose_riho", "oryou"),
    (1818376, 1, 1304, "futagawa_haru", "oryou"),
    (1818395, 1, 2000, "tsubaki_hinako", "oryou"),
    # Amaemi-longing for you-
    (2254574, 13, 211, "kusunoki_iroha", "oryou"),
    (2254574, 212, 378, "akebi_saki", "oryou"),
    (2254574, 379, 558, "kurusu_sakuya", "oryou"),
    (2254574, 563, 750, "kusunoki_iroha", "oryou"),
    (2254574, 751, 927, "akebi_saki", "oryou"),
    (2254574, 928, 1146, "kurusu_sakuya", "oryou"),

    (2254620, 2, 344, "akebi_saki", "oryou"),
    (2254620, 345, 731, "kusunoki_iroha", "oryou"),
    (2254620, 732, 1130, "kurusu_sakuya", "oryou"),
    (2254620, 1131, 1307, "akebi_saki", "oryou"),
    (2254620, 1308, 1495, "kusunoki_iroha", "oryou"),
    (2254620, 1496, 1714, "kurusu_sakuya", "oryou"),

    (2276039, 2, 23, "kusunoki_iroha", "oryou"),
    (2276039, 24, 40, "akebi_saki", "oryou"),
    (2276039, 41, 54, "kurusu_sakuya", "oryou"),
    (2276039, 55, 62, "kusunoki_iroha", "oryou"),
    (2276039, 63, 68, "akebi_saki", "oryou"),
    (2276039, 69, 80, "kurusu_sakuya", "oryou"),
    # ALIA's CARNIVAL!
    (688067, 2, 160, "ousaka_asuha", "mitha"),
    (688067, 172, 290, "saijo_karin", "mitha"),
    (688067, 291, 458, "asamiya_shiina", "nanao naru"),
    (688067, 459, 623, "sakurakouji_tsukuyomi", "nanao naru"),
    (688067, 624, 784, "minase_yuka", "mitha"),

    (1604005, 2, 181, "ousaka_asuha", "mitha"),
    (1604005, 182, 317, "saijo_karin", "mitha"),
    (1604005, 318, 504, "asamiya_shiina", "nanao naru"),
    (1604005, 505, 681, "sakurakouji_tsukuyomi", "nanao naru"),
    (1604005, 682, 860, "minase_yuka", "mitha"),

    (809896, 1, 21, "ousaka_asuha", "mitha"),
    (809896, 22, 33, "asamiya_shiina", "nanao naru"),
    (809896, 34, 52, "sakurakouji_tsukuyomi", "nanao naru"),
    (809896, 53, 69, "saijo_karin", "mitha"),
    (809896, 70, 87, "minase_yuka", "mitha"),

    (938904, 4, 32, "ousaka_asuha", "mitha"),
    (938904, 139, 228, "ousaka_asuha", "mitha"),
    (938904, 339, 434, "saijo_karin", "mitha"),
    (938904, 435, 475, "koikawa_shiho", "nanao naru"),
    (938904, 476, 596, "asamiya_shiina", "nanao naru"),
    (938904, 597, 709, "sakurakouji_tsukuyomi", "nanao naru"),
    (938904, 710, 794, "minase_yuka", "mitha"),

    (938904, 797, 808, "saijo_karin", "mitha"),
    (938904, 809, 821, "asamiya_shiina", "nanao naru"),
    (938904, 856, 874, "sakurakouji_tsukuyomi", "nanao naru"),
    (938904, 875, 886, "minase_yuka", "mitha"),

    (2299688, 1, 451, "ousaka_asuha", "mitha"),
    (2299688, 591, 1106, "saijo_karin", "mitha"),

    (2299695, 1, 235, "koikawa_shiho", "nanao naru"),
    (2299695, 236, 1136, "asamiya_shiina", "nanao naru"),

    (2307617, 1, 618, "sakurakouji_tsukuyomi", "nanao naru"),
    (2307617, 619, 2000, "minase_yuka", "mitha"),

    (2299688, 1, 2000, None, "mitha, nanao naru"),
    (2299695, 1, 2000, None, "mitha, nanao naru"),
    # Haruoto Alice * Gram
    (1081868, 2, 179, "kanzaki_erisa", "mitha"),
    (1081868, 180, 355, "kuonji_kazuha_(harugura)", "mitha"),
    (1081868, 387, 576, "rindou_yaya", "takanae kyourin"),
    (1081868, 577, 740, "fujino_yuki", "nanao naru"),
    (1081868, 741, 911, "shirahane_yuuri", "nanao naru"),

    (1787670, 9, 218, "kanzaki_erisa", "mitha"),
    (1787670, 219, 395, "kuonji_kazuha_(harugura)", "mitha"),
    (1787670, 396, 597, "rindou_yaya", "takanae kyourin"),
    (1787670, 598, 756, "fujino_yuki", "nanao naru"),
    (1787670, 757, 921, "shirahane_yuuri", "nanao naru"),
    # Shirokoi Sakura * Gram
    (1389728, 84, 180, "kanzaki_erisa", "mitha"),
    (1389728, 194, 291, "kuonji_kazuha_(harugura)", "mitha"),
    (1389728, 293, 374, "sema_kozue", "mitha"),
    (1389728, 375, 515, "rindou_yaya", "takanae kyourin"),
    (1389728, 516, 642, "fujino_yuki", "nanao naru"),
    (1389728, 643, 816, "shirahane_yuuri", "nanao naru"),
    (1389728, 824, 884, "rindou_yaya", "takanae kyourin"),
    (1389728, 885, 904, "fujino_yuki", "nanao naru"),
    (1389728, 905, 927, "shirahane_yuuri", "nanao naru"),

    (1741766, 1, 27, "kuonji_kazuha_(harugura)", "mitha"),
    (1741766, 34, 56, "rindou_yaya", "takanae kyourin"),
    (1741766, 65, 93, "fujino_yuki", "nanao naru"),
    (1741766, 94, 116, "kanzaki_erisa", "mitha"),
    (1741766, 124, 166, "shirahane_yuuri", "nanao naru"),

    (1787696, 17, 39, "kanzaki_erisa", "mitha"),
    (1787696, 40, 66, "kuonji_kazuha_(harugura)", "mitha"),
    (1787696, 157, 179, "rindou_yaya", "takanae kyourin"),
    (1787696, 180, 208, "fujino_yuki", "nanao naru"),
    (1787696, 209, 251, "shirahane_yuuri", "nanao naru"),

    (1787696, 554, 650, "kanzaki_erisa", "mitha"),
    (1787696, 664, 761, "kuonji_kazuha_(harugura)", "mitha"),
    (1787696, 762, 843, "sema_kozue", "mitha"),
    (1787696, 844, 984, "rindou_yaya", "takanae kyourin"),
    (1787696, 985, 1110, "fujino_yuki", "nanao naru"),
    (1787696, 1111, 1284, "shirahane_yuuri", "nanao naru"),
    (1787696, 1292, 1350, "rindou_yaya", "takanae kyourin"),
    (1787696, 1351, 1369, "fujino_yuki", "nanao naru"),
    (1787696, 1370, 1392, "shirahane_yuuri", "nanao naru"),

    (2622505, 5, 368, "shirahane_yuuri", "nanao naru"),
    (2622505, 369, 633, "kuonji_kazuha_(harugura)", "mitha"),

    (3533372, 1, 52, "kanzaki_erisa", "mitha"),
    (3533372, 53, 154, "kuonji_kazuha_(harugura)", "mitha"),
    (3533372, 195, 218, "rindou_yaya", "takanae kyourin"),
    (3533372, 219, 359, "fujino_yuki", "nanao naru"),
    (3533372, 360, 458, "shirahane_yuuri", "nanao naru"),

    (3533368, 1, 364, "kanzaki_erisa", "mitha"),
    (3533368, 365, 670, "kuonji_kazuha_(harugura)", "mitha"),
    (3533368, 671, 964, "rindou_yaya", "takanae kyourin"),
    (3533368, 965, 1575, "fujino_yuki", "nanao naru"),
    (3533368, 1576, 1869, "shirahane_yuuri", "nanao naru"),
    (3533369, 109, 216, "sema_kozue", "mitha"),

    (3533370, 1, 364, "kanzaki_erisa", "mitha"),
    (3533370, 365, 721, "kuonji_kazuha_(harugura)", "mitha"),
    (3533370, 722, 1064, "rindou_yaya", "takanae kyourin"),
    (3533370, 1065, 1581, "fujino_yuki", "nanao naru"),
    (3533370, 1582, 1899, "shirahane_yuuri", "nanao naru"),
    (3533371, 1, 27, "sema_kozue", "mitha"),
    # Hana no No ni Saku Utakata no
    (801553, 51, 105, "ouka", "odawara hakone"),
    (801553, 106, 145, "inoue_reina", "odawara hakone"),
    (801553, 146, 190, "yakushi_ryouko", "odawara hakone"),
    (801553, 191, 233, "fujimiya_shione", "odawara hakone"),
    (801553, 234, 268, "kasasagi_shizuku", "odawara hakone"),

    (3509712, 1, 172, "ouka", "odawara hakone"),
    (3509712, 173, 301, "fujimiya_shione", "odawara hakone"),
    (3509712, 302, 409, "inoue_reina", "odawara hakone"),
    (3509712, 410, 573, "yakushi_ryouko", "odawara hakone"),
    (3509712, 574, 669, "kasasagi_shizuku", "odawara hakone"),

    (2251804, 2, 38, "ouka", "odawara hakone"),
    (2251804, 39, 67, "inoue_reina", "odawara hakone"),
    (2251804, 68, 100, "yakushi_ryouko", "odawara hakone"),
    (2251804, 101, 129, "fujimiya_shione", "odawara hakone"),
    (2251804, 130, 153, "kasasagi_shizuku", "odawara hakone"),
    # Annabel Maid Garden
    (1671520, 1, 116, "annabel", "sesena yau"),
    (1671520, 142, 2000, "annabel", "sesena yau"),
    (2254666, 1, 253, "hinohara_tsubomi", "sesena yau"),
    (1942166, 1, 372, "ririum", "sesena yau"),
    (1942166, 406, 412, "ririum", "sesena yau"),
    # Sono Hana ga Saitara, Mata Boku wa Kimi ni Deau
    (1294817, 1, 2000, "nagase_yuki", "nagayama yuunon"),
    (1056055, 49, 249, "lunalight_bake", "nagayama yuunon"),
    # SPIRAL!!
    (1375134, 2, 90, "kadomi_ibarako", "tanihara natsuki"),
    (1375134, 91, 146, "shirokane_mizuki", "nagayama yuunon"),
    (1375134, 147, 210, "ootsu_rose", "ameto yuki"),
    (1375134, 211, 266, "handa_sango", "tanihara natsuki"),
    (1375134, 273, 278, None, "hadumi rio"),

    (1410786, 673, 2000, "kadomi_ibarako", "tanihara natsuki"),
    (1410792, 1, 16, "kadomi_ibarako", "tanihara natsuki"),
    (1410792, 149, 1108, "shirokane_mizuki", "nagayama yuunon"),
    (1410792, 1405, 2000, "ootsu_rose", "ameto yuki"),
    (1410806, 1, 556, "ootsu_rose", "ameto yuki"),
    (1410806, 601, 1752, "handa_sango", "tanihara natsuki"),
    # Amenity's Life
    (1008880, 30, 173, "itano_kanade", "rinks"),
    (1008880, 174, 277, "maho", "rinks"),
    (1008880, 278, 462, "nagamine_mikuri", "rinks"),
    (1008880, 463, 656, "toudou_miki", "rinks"),
    (1008880, 657, 779, "kayama_naru", "rinks"),
    # E School Life
    (1423926, 4, 200, "hanazono_mie", "rinks"),
    (1423926, 201, 306, "sayama_erina", "rinks"),
    (1423926, 307, 519, "tougane_mia", "rinks"),
    (1423926, 520, 734, "suzuyama_misato", "rinks"),
    (1423926, 735, 922, "jinnai_rikako", "rinks"),
    (1423926, 923, 1100, "yoshimura_rin", "rinks"),
    (1423926, 1101, 1192, "karasuma_yuka", "rinks"),
    # Omoide Kakaete Ai ni Koi!!
    (2020714, 5, 433, "sugou_chisato", "rinks"),
    (2020714, 434, 824, "tokorozawa_iku", "rinks"),
    (2020714, 825, 1280, "nishitani_shizuru", "rinks"),
    (2020714, 1281, 1707, "kuriyama_tomari", "rinks"),
    (2020714, 1708, 2000, "harunohara_yuuna", "rinks"),
    (2020719, 2, 131, "harunohara_yuuna", "rinks"),
    (2020719, 188, 257, "sugou_chisato", "rinks"),
    (2020719, 258, 337, "tokorozawa_iku", "rinks"),
    (2020719, 338, 437, "nishitani_shizuru", "rinks"),
    (2020719, 438, 534, "kuriyama_tomari", "rinks"),
    (2020719, 535, 612, "harunohara_yuuna", "rinks"),
    # Making * Lovers
    (2204943, 2, 130, "takanashi_ako", "taniyama-san"),
    (2204943, 131, 286, "kitaooji_karen", "taniyama-san"),
    (2204943, 287, 444, "tsukino_mashiro", "taniyama-san"),
    (2204943, 445, 574, "kanome_reina", "taniyama-san"),
    (2204943, 575, 716, "naruse_saki", "taniyama-san"),
    (2204943, 718, 795, "takanashi_ako", "taniyama-san"),
    (2204943, 796, 865, "kitaooji_karen", "taniyama-san"),
    (2204943, 866, 939, "tsukino_mashiro", "taniyama-san"),
    (2204943, 940, 999, "kanome_reina", "taniyama-san"),
    (2204943, 1000, 1077, "naruse_saki", "taniyama-san"),

    (1146573, 2, 125, "takanashi_ako", "taniyama-san"),
    (1146573, 126, 281, "kitaooji_karen", "taniyama-san"),
    (1146573, 282, 439, "tsukino_mashiro", "taniyama-san"),
    (1146573, 440, 569, "kanome_reina", "taniyama-san"),
    (1146573, 570, 712, "naruse_saki", "taniyama-san"),

    (1215973, 3, 80, "takanashi_ako", "taniyama-san"),
    (1215973, 81, 145, "kitaooji_karen", "taniyama-san"),
    (1215973, 146, 218, "naruse_saki", "taniyama-san"),
    (1245550, 3, 76, "tsukino_mashiro", "taniyama-san"),
    (1245550, 77, 136, "kanome_reina", "taniyama-san"),

    (1603013, 3, 131, "takanashi_ako", "taniyama-san"),
    (1603013, 132, 287, "kitaooji_karen", "taniyama-san"),
    (1603013, 288, 445, "tsukino_mashiro", "taniyama-san"),
    (1603013, 446, 575, "kanome_reina", "taniyama-san"),
    (1603013, 576, 717, "naruse_saki", "taniyama-san"),
    (1603013, 718, 788, "takanashi_ako", "taniyama-san"),
    (1603013, 789, 857, "kitaooji_karen", "taniyama-san"),
    (1603013, 858, 939, "tsukino_mashiro", "taniyama-san"),
    (1603013, 940, 1001, "kanome_reina", "taniyama-san"),
    (1603013, 1002, 1053, "naruse_saki", "taniyama-san"),

    (2204943, 2, 130, "takanashi_ako", "taniyama-san"),
    (2204943, 131, 286, "kitaooji_karen", "taniyama-san"),
    (2204943, 287, 444, "tsukino_mashiro", "taniyama-san"),
    (2204943, 445, 574, "kanome_reina", "taniyama-san"),
    (2204943, 575, 716, "naruse_saki", "taniyama-san"),
    (2204943, 718, 795, "takanashi_ako", "taniyama-san"),
    (2204943, 796, 865, "kitaooji_karen", "taniyama-san"),
    (2204943, 866, 939, "tsukino_mashiro", "taniyama-san"),
    (2204943, 940, 999, "kanome_reina", "taniyama-san"),
    (2204943, 1000, 1077, "naruse_saki", "taniyama-san"),

    (1340253, 2, 125, "takanashi_ako", "taniyama-san"),
    (1340253, 126, 281, "kitaooji_karen", "taniyama-san"),
    (1340253, 282, 439, "tsukino_mashiro", "taniyama-san"),
    (1340253, 440, 569, "kanome_reina", "taniyama-san"),
    (1340253, 570, 711, "naruse_saki", "taniyama-san"),

    # Sugar*Style
    (1354599, 7, 117, "kusunoki_hare", "taniyama-san"),
    (1354599, 118, 235, "morisumi_ichika", "taniyama-san"),
    (1354599, 236, 342, "fuyutsuki_kaname", "taniyama-san"),
    (1354599, 343, 452, "minami_mao", "taniyama-san"),

    (1439306, 4, 55, "kusunoki_hare", "taniyama-san"),
    (1439306, 56, 90, "morisumi_ichika", "taniyama-san"),
    (1439306, 91, 128, "fuyutsuki_kaname", "taniyama-san"),
    (1439306, 129, 163, "minami_mao", "taniyama-san"),

    (1497893, 1, 14, "morisumi_ichika", "taniyama-san"),

    (1900457, 7, 117, "kusunoki_hare", "taniyama-san"),
    (1900457, 118, 235, "morisumi_ichika", "taniyama-san"),
    (1900457, 236, 342, "fuyutsuki_kaname", "taniyama-san"),
    (1900457, 343, 452, "minami_mao", "taniyama-san"),

    # Harem Kingdom
    (1579777, 284, 339, "kiki", "taniyama-san"),
    (1579777, 340, 402, "marrow", "taniyama-san"),
    (1579777, 403, 462, "sophia", "taniyama-san"),
    (1579777, 463, 520, "charlone", "taniyama-san"),
    (1579777, 521, 582, "hikari_(harem_kingdom)", "taniyama-san"),

    (1742694, 13, 46, "kiki", "taniyama-san"),
    (1742694, 47, 96, "sophia", "taniyama-san"),
    (1742694, 97, 131, "hikari_(harem_kingdom)", "taniyama-san"),

    (1787981, 13, 62, "marrow", "taniyama-san"),
    (1787981, 63, 110, "charlone", "taniyama-san"),
    # 1/1 Kareshi Kanojo
    (2152033, 2, 106, "fujigaya_eimi", "taniyama-san"),
    (2152033, 107, 217, "colette_carlini", "taniyama-san"),
    (2152033, 218, 333, "izumi_taeko", "taniyama-san"),
    (2152033, 334, 438, "seimiya_chizu", "taniyama-san"),

    (2454924, 3, 57, "fujigaya_eimi", "taniyama-san"),
    (2454924, 58, 118, "colette_carlini", "taniyama-san"),
    (2454924, 119, 168, "izumi_taeko", "taniyama-san"),
    (2454924, 169, 227, "seimiya_chizu", "taniyama-san"),

    (2454927, 10, 92, "fujigaya_eimi", "taniyama-san"),
    (2454927, 93, 165, "colette_carlini", "taniyama-san"),
    (2454927, 166, 228, "izumi_taeko", "taniyama-san"),
    (2454927, 229, 300, "seimiya_chizu", "taniyama-san"),
    # Kanojo * Step
    (981370, 6, 141, "yanagi_asuka", None),
    (981370, 143, 267, "serizawa_kuon", None),
    (981370, 268, 394, "kisaragi_noe", None),
    (981370, 395, 517, "kano_shiina", None),
    # Happy Weekend
    (3415528, 2, 546, "izumi_akina", None),
    (3415528, 547, 1588, "hanasaki_koharu", None),
    (3415529, 2, 700, "shiina_yuki", None),
    # Kanojo wa Ecchi de Midara na Hentai
    (722918, 17, 179, "yamashina_kaori", "sorai shinya"),
    (722918, 180, 346, "jinguuji_reika", "kaniya shiku"),
    (722918, 347, 534, "katou_riko", "kannon ouji"),
    (722918, 535, 691, "kinosaki_yoshino", "sorai shinya"),

    (729988, 2, 49, "shijou_hinako", "kannon ouji"),

    (730780, 2, 241, "yamashina_kaori", "sorai shinya"),
    (730780, 242, 497, "shijou_hinako", "kannon ouji"),
    (730780, 498, 752, "kinosaki_yoshino", "sorai shinya"),
    (730780, 753, 1016, "katou_riko", "kannon ouji"),
    (730780, 1017, 1344, "jinguuji_reika", "kaniya shiku"),
    # Soshite Hatsukoi ga Imouto ni Nar
    (939815, 166, 239, "kawatsu_tsubasa", "sorai shinya"),
    (939815, 240, 304, "miyamoto_yuka", "sorai shinya"),
    (939815, 305, 368, "tanaka_neneko", "sorai shinya"),
    (939815, 369, 441, "tokitani_shinobu", "sorai shinya"),
    # Shunki Gentei Poco a Poco!
    (436844, 59, 136, "ninomae_sakura", "kazami haruki"),
    (436844, 138, 225, "nonomiya_ai", "takoyaki"),
    (436844, 226, 297, "yuuki_natsumi", "mutou kurihito"),
    (436844, 412, 508, "ninomae_sakura", "kazami haruki"),
    (436844, 510, 597, "nonomiya_ai", "takoyaki"),
    (436844, 598, 669, "yuuki_natsumi", "mutou kurihito"),

    (1023742, 19, 135, "ninomae_sakura", "kazami haruki"),
    (1023742, 137, 224, "nonomiya_ai", "takoyaki"),
    (1023742, 225, 296, "yuuki_natsumi", "mutou kurihito"),
    # Love La Bride
    (586926, 2, 100, "sakuranomori_misaki", "takano yuki"),
    (586926, 106, 188, "yuki_nao", "mutou kurihito"),
    (586926, 189, 280, "omigawa_hitomi", "takano yuki"),
    (586926, 281, 378, "mikami_haruka", "mutou kurihito"),
    (586926, 379, 455, "sasha_(love_ra_bride)", "takano yuki"),
    # Ojousama wa Gokigen Naname
    (595808, 16, 168, "sakurazaki_hana", "mutou kurihito"),
    (595808, 169, 341, "naname_nanami", "mutou kurihito"),
    (595808, 342, 499, "hibiki_otoha", "mutou kurihito"),
    (595808, 500, 648, "yukinokouji_shiaya", "mutou kurihito"),
    (595808, 649, 812, "ichimine_touka", "mutou kurihito"),

    (1512898, 13, 37, "naname_nanami", "mutou kurihito"),
    (1512898, 46, 151, "sakurazaki_hana", "mutou kurihito"),
    (1512898, 152, 254, "naname_nanami", "mutou kurihito"),
    (1512898, 255, 351, "hibiki_otoha", "mutou kurihito"),
    (1512898, 352, 440, "yukinokouji_shiaya", "mutou kurihito"),
    (1512898, 441, 546, "ichimine_touka", "mutou kurihito"),
    (1512898, 720, 959, "hibiki_otoha", "mutou kurihito"),
    (1512898, 960, 1143, "sakurazaki_hana", "mutou kurihito"),
    (1512898, 1144, 1335, "yukinokouji_shiaya", "mutou kurihito"),
    (1512898, 1336, 1521, "naname_nanami", "mutou kurihito"),
    (1512898, 1652, 1843, "ichimine_touka", "mutou kurihito"),

    (645105, 2, 49, "naname_nanami", "mutou kurihito"),
    # Imouto no Katachi
    (1003292, 2, 131, "sena_miyuki", "mutou kurihito"),
    (1003292, 132, 221, "suzunomiya_mayuki", "mutou kurihito"),
    (1003292, 222, 361, "sumeragi_ayaka", "hashimoto takashi"),
    (1003292, 362, 459, "mima_chimari", "kodamasawa"),
    (1003292, 460, 505, "meta_(imouto_no_katachi)", "mutou kurihito"),

    (706412, 2, 120, "sena_miyuki", "mutou kurihito"),
    (706412, 121, 188, "suzunomiya_mayuki", "mutou kurihito"),
    (706412, 189, 258, "sumeragi_ayaka", "hashimoto takashi"),
    (706412, 259, 363, "mima_chimari", "kodamasawa"),
    (706412, 429, 493, "meta_(imouto_no_katachi)", "mutou kurihito"),
    # －HOSHI ORI ★ YUME MIRAI－
    (723519, 2, 2000, "yukimura_touko", "iizuki tasuku"),

    (723566, 3, 537, "shinozaki_marika", "koizumi amane"),
    (723566, 538, 973, "okihara_misa", "koizumi amane"),
    (723566, 974, 1684, "segawa_natsuki", "akino subaru"),
    (723566, 1685, 2000, "narusawa_rikka", "akino subaru"),
    (723567, 2, 721, "narusawa_rikka", "akino subaru"),
    (723567, 722, 1265, "ousaka_sora", "mutou kurihito"),
    (723567, 1270, 2000, "yukimura_touko", "iizuki tasuku"),

    (2120701, 2, 634, "shinozaki_marika", "koizumi amane"),
    (2120701, 635, 1005, "okihara_misa", "koizumi amane"),

    (1978381, 2, 668, "segawa_natsuki", "akino subaru"),
    (1978381, 669, 1799, "narusawa_rikka", "akino subaru"),

    (1978412, 2, 617, "ousaka_sora", "mutou kurihito"),
    (1978412, 618, 1429, "yukimura_touko", "iizuki tasuku"),

    (3371033, 1, 2000, "ousaka_sora", "mutou kurihito"),
    (3372545, 1, 2000, "shinozaki_marika", "koizumi amane"),
    # Otome ga Irodoru Koi no Essence
    (918460, 3, 158, "natsume_akari", "asami asami"),
    (918460, 190, 378, "chihara_noa", "mutou kurihito"),
    (918460, 379, 593, "amagi_sakuya", "shona mitsuishi"),
    (918460, 594, 812, "makise_satsuki", "zinno"),
    (918460, 813, 993, "hyoudou_serika", "zinno"),

    (998656, 8, 53, "natsume_akari", "asami asami"),
    (998656, 74, 131, "chihara_noa", "mutou kurihito"),
    (998656, 132, 211, "amagi_sakuya", "shona mitsuishi"),
    (998656, 212, 295, "makise_satsuki", "zinno"),
    (998656, 296, 358, "hyoudou_serika", "zinno"),

    (1946978, 9, 54, "natsume_akari", "asami asami"),
    (1946978, 76, 132, "chihara_noa", "mutou kurihito"),
    (1946978, 133, 212, "amagi_sakuya", "shona mitsuishi"),
    (1946978, 213, 295, "makise_satsuki", "zinno"),
    (1946978, 296, 361, "hyoudou_serika", "zinno"),
    # Giniro, Haruka
    (969937, 3, 437, "bethly_rose_daisley", "koizumi amane"),
    (969937, 438, 1065, "aoi_hinata", "akino subaru"),
    (969937, 1066, 1665, "nashiro_momiji", "koizumi amane"),
    (970026, 1, 1340, "kisaragi_mizuha", "akino subaru"),
    (970026, 1341, 1745, "niimi_yuzuki", "mutou kurihito"),
    # Ojou-sama wa Sunao ni Narenai
    (1078663, 59, 146, "hiiragi_erika", "machimura komori"),
    (1078663, 147, 233, "hikami_kuon", "naruse hirofumi"),
    (1078663, 234, 326, "jinguuji_miku", "noba"),
    (1078663, 327, 408, "mibu_natsuki_(ojou-sama_wa_sunao_ni_narenai)", "yashima takahiro"),
    (1078663, 409, 488, "konoe_rikka", "mutou kurihito"),

    (1204243, 3, 46, "hiiragi_erika", "machimura komori"),
    (1204243, 47, 90, "hikami_kuon", "naruse hirofumi"),
    (1204243, 91, 135, "jinguuji_miku", "noba"),
    (1204243, 136, 187, "mibu_natsuki_(ojou-sama_wa_sunao_ni_narenai)", "yashima takahiro"),
    (1204243, 188, 234, "konoe_rikka", "mutou kurihito"),
    # Kanojo wa Tenshi de Imouto de
    # Otome ga Musubu Tsukiyo no Kirameki
    (1322644, 53, 100, "sakura_kokoro", "mizuki yuuma"),
    (1322644, 127, 213, "fujisaki_nagisa", "kurasawa moko"),
    (1322644, 217, 286, "shijou_ran", "mutou kurihito"),
    (1322644, 308, 396, "reizei_shizune", "sesena yau"),
    (1322644, 397, 493, "shinomori_suzuka", "kurasawa moko"),

    (1328171, 1, 144, "shinomori_suzuka", "kurasawa moko"),
    (1328171, 145, 296, "shijou_ran", "mutou kurihito"),
    (1328171, 411, 464, "sakura_kokoro", "mizuki yuuma"),
    (1328171, 519, 734, "reizei_shizune", "sesena yau"),
    (1328171, 863, 1006, "fujisaki_nagisa", "kurasawa moko"),
    (1328171, 1205, 1240, "shijou_ran", "mutou kurihito"),

    (1438003, 42, 66, "sakura_kokoro", "mizuki yuuma"),
    (1438003, 87, 149, "fujisaki_nagisa", "kurasawa moko"),
    (1438003, 151, 205, "shijou_ran", "mutou kurihito"),
    (1438003, 206, 257, "reizei_shizune", "sesena yau"),
    (1438003, 258, 315, "shinomori_suzuka", "kurasawa moko"),

    (1438041, 39, 64, "sakura_kokoro", "mizuki yuuma"),
    (1438041, 85, 146, "fujisaki_nagisa", "kurasawa moko"),
    (1438041, 147, 201, "shijou_ran", "mutou kurihito"),
    (1438041, 202, 253, "reizei_shizune", "sesena yau"),
    (1438041, 254, 311, "shinomori_suzuka", "kurasawa moko"),
    # Tsuki no Kanata de Aimashou
    (1438905, 4, 262, "araya_touka", "koizumi amane"),
    (1438905, 263, 610, "hishiki_uguisu", "koikeya"),
    (1438905, 611, 1061, "sakura_rein", "akino subaru"),
    (1438905, 1064, 1347, "kurawashi_seira", "mutou kurihito"),
    (1438905, 1348, 1599, "misaki_kanna", "koikeya"),
    (1438905, 1600, 1809, "matsumiya_kiriko", "akino subaru"),
    (1438905, 1810, 2000, "tsukigahora_kirari", "koizumi amane"),

    (1648777, 1, 1345, "sakura_rein", "akino subaru"),
    # Yuukoku no Femme Fatale
    (3073032, 2, 74, "marigold", "mutou kurihito"),
    (3073032, 75, 134, "rosemary", "mutou kurihito"),
    (3073032, 135, 199, "lavender", "mutou kurihito"),
    (3073032, 200, 273, "black_lily", "mutou kurihito"),

    (3084850, 6, 153, "marigold", "mutou kurihito"),
    (3084850, 154, 285, "rosemary", "mutou kurihito"),
    (3084850, 286, 427, "lavender", "mutou kurihito"),
    (3084850, 428, 565, "black_lily", "mutou kurihito"),

    (3101255, 1, 147, "marigold", "mutou kurihito"),
    (3101255, 148, 280, "rosemary", "mutou kurihito"),
    (3101255, 281, 422, "lavender", "mutou kurihito"),
    (3101255, 423, 560, "black_lily", "mutou kurihito"),

    (3101372, 73, 224, "marigold", "mutou kurihito"),
    (3101372, 225, 362, "lavender", "mutou kurihito"),
    (3101372, 363, 560, "black_lily", "mutou kurihito"),
    (3101372, 561, 686, "rosemary", "mutou kurihito"),
    (3101372, 841, 992, "marigold", "mutou kurihito"),
    (3101372, 993, 1130, "lavender", "mutou kurihito"),
    (3101372, 1131, 1328, "black_lily", "mutou kurihito"),
    (3101372, 1329, 1454, "rosemary", "mutou kurihito"),

    (3101373, 73, 224, "marigold", "mutou kurihito"),
    (3101373, 225, 362, "lavender", "mutou kurihito"),
    (3101373, 363, 560, "black_lily", "mutou kurihito"),
    (3101373, 561, 686, "rosemary", "mutou kurihito"),
    (3101373, 841, 992, "marigold", "mutou kurihito"),
    (3101373, 993, 1130, "lavender", "mutou kurihito"),
    (3101373, 1131, 1328, "black_lily", "mutou kurihito"),
    (3101373, 1329, 1454, "rosemary", "mutou kurihito"),

    (3101372, 1, 2000, None, "mutou kurihito"),
    (3101373, 1, 2000, None, "mutou kurihito"),
    # Mugen no Tír na nÓg
    (3163973, 2, 69, "iris_murphy", "ameto yuki"),
    (3163973, 70, 105, "yusa_karen", "nanohana kohina"),
    (3163973, 106, 150, "midorisawa_aoi", "ameto yuki"),
    (3163973, 151, 193, "hoshiko_sakuya", "nanohana kohina"),
    (3163973, 194, 210, "percy_portion", "ameto yuki"),
    (3163973, 249, 341, "iris_murphy", "ameto yuki"),
    (3163973, 342, 433, "yusa_karen", "nanohana kohina"),
    (3163973, 434, 520, "midorisawa_aoi", "ameto yuki"),
    (3163973, 521, 607, "hoshiko_sakuya", "nanohana kohina"),
    (3163973, 608, 650, "percy_portion", "ameto yuki"),
    (3163973, 651, 851, "iris_murphy", "ameto yuki"),

    (3424828, 1, 440, "iris_murphy", "ameto yuki"),
    (3424828, 441, 674, "yusa_karen", "nanohana kohina"),
    (3424828, 675, 1082, "midorisawa_aoi", "ameto yuki"),
    (3424828, 1083, 1322, "hoshiko_sakuya", "nanohana kohina"),
    (3424828, 1323, 1552, "percy_portion", "ameto yuki"),
    # Onii-chan Chou Suki 99 Jikan Ecchi Shiyo!
    (1922003, 3, 162, "mizuki", "akino subaru"),
    (1922003, 163, 322, "tsumugi", "akino subaru"),
    # White Blue
    (1438240, 1, 271, "uekusa_hibari", "ayase hazuki"),
    (1438240, 272, 304, "fushimi_angelia", "ayase hazuki"),
    (1438240, 305, 330, "uekusa_hibari", "ayase hazuki"),
    (1438240, 331, 337, "fushimi_angelia", "ayase hazuki"),
    (1438240, 338, 363, "uekusa_hibari", "ayase hazuki"),
    (1438240, 364, 400, "fushimi_angelia", "ayase hazuki"),

    (1439401, 2, 81, "uekusa_hibari", "ayase hazuki"),
    (1439401, 82, 111, "fushimi_angelia", "ayase hazuki"),
    (1439401, 154, 233, "uekusa_hibari", "ayase hazuki"),
    (1439401, 234, 263, "fushimi_angelia", "ayase hazuki"),

    (2263210, 1, 287, "uekusa_hibari", "ayase hazuki"),
    (2263210, 288, 399, "fushimi_angelia", "ayase hazuki"),
    (2263210, 403, 435, "uekusa_hibari, fushimi_angelia", "ayase hazuki"),
    # oi Suru Kanojo no Bukiyou na Butai
    (761970, 8, 387, "togawa_mayuu", "kantoku"),
    (761970, 388, 773, "nanase_sena", "kantoku"),
    (761970, 764, 1129, "alice_third_macy", "kantoku"),
    (761970, 1130, 1455, "chiyoda_momoka", "kantoku"),

    (1311837, 8, 387, "togawa_mayuu", "kantoku"),
    (1311837, 388, 773, "nanase_sena", "kantoku"),
    (1311837, 764, 1129, "alice_third_macy", "kantoku"),
    (1311837, 1130, 1455, "chiyoda_momoka", "kantoku"),

    (1163452, 1, 140, "togawa_mayuu", "kantoku"),
    (1163452, 141, 300, "nanase_sena", "kantoku"),
    (1163452, 301, 412, "alice_third_macy", "kantoku"),
    (1163452, 413, 524, "chiyoda_momoka", "kantoku"),
    # your diary
    (881468, 4, 168, "yua_(your_diary)", "kantoku"),
    (881468, 169, 333, "ayase_sayuki", "kantoku"),
    (881468, 334, 488, "minagawa_yuuhi", "kantoku"),
    (881468, 489, 657, "hirosaki_kanade", "kantoku"),

    (766621, 3, 179, "yua_(your_diary)", "kantoku"),
    (766621, 180, 357, "ayase_sayuki", "kantoku"),
    (766621, 358, 523, "minagawa_yuuhi", "kantoku"),
    (766621, 524, 703, "hirosaki_kanade", "kantoku"),

    (3425989, 1, 270, "yua_(your_diary)", "kantoku"),
    (3425989, 271, 495, "ayase_sayuki", "kantoku"),
    (3425989, 496, 817, "minagawa_yuuhi", "kantoku"),
    (3425989, 818, 1048, "hirosaki_kanade", "kantoku"),

    (3425956, 1, 270, "yua_(your_diary)", "kantoku"),
    (3425956, 271, 495, "ayase_sayuki", "kantoku"),
    (3425956, 496, 863, "minagawa_yuuhi", "kantoku"),
    (3425956, 864, 1139, "hirosaki_kanade", "kantoku"),
    # Kami-sama no You na Kimi e
    (1596944, 1, 158, "tsukuyomi_(kamikimi)", "kantoku"),
    (1596944, 159, 321, "lana_liddell-hart", "kantoku"),
    (1596944, 354, 436, "asakura_kirika", "kantoku"),
    (1596944, 437, 516, "kokuhou_rein", "kantoku"),
    (1596944, 517, 604, "sophia_liddell-hart", "kantoku"),
    (1596944, 605, 673, "kannazuki_airi", "kantoku"),

    (1733086, 2, 206, "tsukuyomi_(kamikimi)", "kantoku"),
    (2154465, 94, 298, "tsukuyomi_(kamikimi)", "kantoku"),
    (2154465, 299, 788, "lana_liddell-hart", "kantoku"),
    (2154465, 789, 950, "asakura_kirika", "kantoku"),
    (2154465, 951, 1001, "kokuhou_rein", "kantoku"),
    (2154465, 1002, 1055, "sophia_liddell-hart", "kantoku"),
    (2154465, 1056, 1153, "kannazuki_airi", "kantoku"),

    (2180680, 135, 178, "tsukuyomi_(kamikimi)", "kantoku"),
    (2180680, 179, 231, "lana_liddell-hart", "kantoku"),
    (2180680, 232, 255, "asakura_kirika", "kantoku"),
    (2180680, 256, 280, "kokuhou_rein", "kantoku"),
    (2180680, 281, 306, "sophia_liddell-hart", "kantoku"),
    (2180680, 307, 391, "sonomura_hazuki", "kantoku"),

    (3425997, 1, 205, "tsukuyomi_(kamikimi)", "kantoku"),
    (3425997, 206, 695, "lana_liddell-hart", "kantoku"),
    (3425997, 696, 857, "asakura_kirika", "kantoku"),
    (3425997, 858, 908, "kokuhou_rein", "kantoku"),
    (3425997, 909, 962, "sophia_liddell-hart", "kantoku"),
    (3425997, 963, 1060, "kannazuki_airi", "kantoku"),
    (3425997, 1089, 1504, "sonomura_hazuki", "kantoku"),
    # Mamiya-kun-chi no Itsutsugo Jijou
    (908928, 3, 155, "mamiya_yakumo", "kanekiyo miwa"),
    (908928, 156, 309, "mamiya_moeri", "kanekiyo miwa"),
    (908928, 310, 477, "mamiya_tamane", "kanekiyo miwa"),
    (908928, 478, 659, "mamiya_nekoto", "kanekiyo miwa"),
    (908928, 660, 820, "shijouin_ririka", "kanekiyo miwa"),
    (908928, 821, 859, "mamiya_kyouka", "kanekiyo miwa"),
    (908928, 1, 2000, None, "kanekiyo miwa"),

    (1408222, 1, 105, "mamiya_yakumo", "kanekiyo miwa"),
    (1408222, 106, 210, "mamiya_moeri", "kanekiyo miwa"),
    (1408222, 211, 329, "mamiya_tamane", "kanekiyo miwa"),
    (1408222, 330, 449, "mamiya_nekoto", "kanekiyo miwa"),
    (1408222, 450, 554, "shijouin_ririka", "kanekiyo miwa"),
    (1408222, 555, 624, "mamiya_kyouka", "kanekiyo miwa"),

    (3426048, 1, 105, "mamiya_yakumo", "kanekiyo miwa"),
    (3426048, 106, 210, "mamiya_moeri", "kanekiyo miwa"),
    (3426048, 211, 329, "mamiya_tamane", "kanekiyo miwa"),
    (3426048, 330, 449, "mamiya_nekoto", "kanekiyo miwa"),
    (3426048, 450, 554, "shijouin_ririka", "kanekiyo miwa"),
    (3426048, 555, 624, "mamiya_kyouka", "kanekiyo miwa"),
    # Kurano-kunchi no Futago Jijou
    (1335358, 2, 231, "kurano_izumi", "kanekiyo miwa"),
    (1335358, 232, 423, "kurano_mikoto", "kanekiyo miwa"),
    (1335358, 424, 711, "kurano_tomoka", "kanekiyo miwa"),
    (1335358, 712, 959, "kurano_yae", "kanekiyo miwa"),
    (1335358, 960, 1229, "kurano_ema", "kanekiyo miwa"),
    (1335358, 1388, 1398, "kurano_mikoto", "kanekiyo miwa"),
    (1335358, 1, 2000, None, "kanekiyo miwa"),

    (3425981, 1, 105, "kurano_izumi", "kanekiyo miwa"),
    (3425981, 106, 240, "kurano_mikoto", "kanekiyo miwa"),
    (3425981, 241, 375, "kurano_tomoka", "kanekiyo miwa"),
    (3425981, 376, 480, "kurano_yae", "kanekiyo miwa"),
    (3425981, 481, 585, "kurano_ema", "kanekiyo miwa"),
    # Yurameku Kokoro ni Michita Sekai de, Kimi no Yume to Yokubou wa Kanau ka
    (1146570, 4, 168, "amatori_sumire", "kakao"),
    (1146570, 169, 313, "fushimi_tomoe", "kakao"),
    (1146570, 314, 478, "mizusaki_tsukino", "kanekiyo miwa"),
    (1146570, 479, 638, "yukishiro_himari", "kanekiyo miwa"),

    (1405214, 1, 130, "amatori_sumire", "kakao"),
    (1405214, 131, 260, "fushimi_tomoe", "kakao"),
    (1405214, 261, 520, "mizusaki_tsukino", "kanekiyo miwa"),
    (1405214, 521, 664, "yukishiro_himari", "kanekiyo miwa"),

    (3426050, 1, 130, "amatori_sumire", "kakao"),
    (3426050, 131, 260, "fushimi_tomoe", "kakao"),
    (3426050, 261, 520, "mizusaki_tsukino", "kanekiyo miwa"),
    (3426050, 521, 664, "yukishiro_himari", "kanekiyo miwa"),
    # Umi to Yuki no Cyan Blue
    (1900069, 5, 149, "aono_nana", "kanekiyo miwa"),
    (1900069, 150, 306, "sumiki_kotoha", "kanekiyo miwa"),
    (1900069, 307, 473, "nabatake_inaba", "kanekiyo miwa"),
    (1900069, 474, 615, "hagino_yume", "kurasawa moko"),
    (1900069, 616, 766, "matsuki_ira", "tsurukonnyaku"),
    (1900069, 778, 785, "aono_nana", "kanekiyo miwa"),

    (3426078, 1, 208, "aono_nana", "kanekiyo miwa"),
    (3426078, 209, 416, "sumiki_kotoha", "kanekiyo miwa"),
    (3426078, 417, 668, "nabatake_inaba", "kanekiyo miwa"),
    (3426078, 669, 932, "hagino_yume", "kurasawa moko"),
    (3426078, 933, 1244, "matsuki_ira", "tsurukonnyaku"),
    (3426078, 1245, 1295, "utsuki_gunjou", "kanekiyo miwa"),
    # Same to Ikiru Nanokakan
    (2384657, 5, 54, "kuuko_(same_to_ikiru_nanokakan)", "tsurukonnyaku"),
    (2384657, 57, 77, "tenjiku_amane", "yusano"),
    (2384657, 81, 109, "tenjiku_amane", "yusano"),
    (2384657, 110, 131, "hoshiro_remi", "tsurukonnyaku"),
    (2384657, 133, 137, "hoshiro_remi", "tsurukonnyaku"),
    (2384657, 138, 176, "funabori_onee-san", "tsurukonnyaku"),
    (2384657, 177, 191, "yoshikiri_tooka", "kurasawa moko"),
    (2384657, 192, 215, "nemuribuka_maruri", "kurot"),
    (2384657, 228, 345, "kuuko_(same_to_ikiru_nanokakan)", "tsurukonnyaku"),
    (2384657, 346, 453, "tenjiku_amane", "yusano"),
    (2384657, 454, 568, "hoshiro_remi", "tsurukonnyaku"),
    (2384657, 569, 640, "funabori_onee-san", "tsurukonnyaku"),
    (2384657, 641, 693, "yoshikiri_tooka", "kurasawa moko"),
    (2384657, 694, 723, "nemuribuka_maruri", "kurot"),

    (3424965, 1, 382, "kuuko_(same_to_ikiru_nanokakan)", "tsurukonnyaku"),
    (3424965, 383, 802, "tenjiku_amane", "yusano"),
    (3424965, 803, 1354, "hoshiro_remi", "tsurukonnyaku"),
    (3424966, 1, 246, "funabori_onee-san", "tsurukonnyaku"),
    (3424966, 247, 414, "yoshikiri_tooka", "kurasawa moko"),
    (3424966, 415, 736, "nemuribuka_maruri", "kurot"),

    # Natsu no Owari
    (2310933, 1, 379, "moi", "u35"),
    (3425864, 1, 2000, "moi", "u35"),
    # natsu no Ame
    (636926, 3, 70, "segawa_rikako", "kanekiyo miwa"),
    (636926, 71, 103, "miyazawa_midori", "kantoku"),
    (636926, 104, 143, "itou_hinako", "kantoku"),
    (636926, 144, 179, "shinooka_misa", "kanekiyo miwa"),

    (465688, 1, 69, "segawa_rikako", "kanekiyo miwa"),
    (465688, 70, 109, "miyazawa_midori", "kantoku"),
    (465688, 110, 140, "itou_hinako", "kantoku"),
    (465688, 141, 174, "shinooka_misa", "kanekiyo miwa"),

    (3426003, 1, 231, "segawa_rikako", "kanekiyo miwa"),
    (3426003, 232, 375, "miyazawa_midori", "kantoku"),
    (3426003, 376, 490, "itou_hinako", "kantoku"),
    (3426003, 491, 558, "shinooka_misa", "kanekiyo miwa"),
    # Onaji Class no Idol-san
    (1491228, 4, 158, "mishima_akari", "unasaka"),
    (1491228, 159, 316, "yuuki_wako", "shona mitsuishi"),
    (1491228, 318, 474, "takanashi_ei", "sawayaka samehada"),
    (1491228, 475, 536, "naruse_kanae", "niro"),

    (1498992, 1, 216, "mishima_akari", "unasaka"),
    (1498992, 217, 432, "yuuki_wako", "shona mitsuishi"),
    (1498992, 433, 640, "takanashi_ei", "sawayaka samehada"),
    (1498992, 641, 730, "naruse_kanae", "niro"),
    # Uchi wa Mou, Enki Dekinai.
    (2070790, 5, 166, "miyamura_miku", "sawayaka samehada"),
    (2070790, 167, 303, "miyamura_karin", "sawayaka samehada"),
    (2070790, 304, 457, "suzumoto_chisa", "unasaka"),
    (2070790, 458, 608, "sakakibara_ai", "sawayaka samehada"),
    # Hibikino-san-chi wa Erogeya-san!
    (1785639, 6, 163, "hibikino_tsumugi", "sawayaka samehada"),
    (1785639, 164, 311, "hibikino_yukari", "sawayaka samehada"),
    (1785639, 312, 467, "hibikino_yui", "sawayaka samehada"),
    (1785639, 468, 647, "hayami_shizuno", "unasaka"),
    (1785639, 648, 743, "honomi_youko", "mizuno sao"),

    (1809787, 1, 10, "hibikino_tsumugi", "sawayaka samehada"),
    (1809787, 11, 20, "hibikino_yukari", "sawayaka samehada"),
    (1809787, 21, 29, "hibikino_yui", "sawayaka samehada"),
    (1809787, 30, 39, "hayami_shizuno", "unasaka"),
    (1809787, 40, 49, "honomi_youko", "mizuno sao"),
    # Irotoridori no Sekai
    (392833, 108, 253, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (392833, 254, 361, "shikishima_kyou", "natsume eri"),
    (392833, 362, 515, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (392833, 561, 690, "nikaidou_shinku", "shida kazuhiro"),
    (392833, 691, 779, "toumine_tsukasa", "gt"),

    (2342163, 114, 261, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2342163, 262, 369, "shikishima_kyou", "natsume eri"),
    (2342163, 370, 519, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2342163, 520, 649, "nikaidou_shinku", "shida kazuhiro"),
    (2342163, 650, 738, "toumine_tsukasa", "gt"),

    (2870141, 1, 58, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2870141, 59, 115, "shikishima_kyou", "natsume eri"),
    (2870141, 116, 185, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2870141, 186, 218, "nikaidou_shinku", "shida kazuhiro"),
    (2870141, 219, 263, "toumine_tsukasa", "gt"),

    (3362955, 112, 259, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (3362955, 260, 367, "shikishima_kyou", "natsume eri"),
    (3362955, 368, 521, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (3362955, 567, 696, "nikaidou_shinku", "shida kazuhiro"),
    (3362955, 697, 785, "toumine_tsukasa", "gt"),

    (522628, 124, 249, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (522628, 250, 352, "shikishima_kyou", "natsume eri"),
    (522628, 353, 450, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (522628, 487, 694, "nikaidou_shinku", "shida kazuhiro"),
    (522628, 695, 794, "toumine_tsukasa", "gt"),

    (2342188, 125, 250, "minami_kana_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2342188, 251, 353, "shikishima_kyou", "natsume eri"),
    (2342188, 354, 451, "kisaragi_mio_(irotoridori_no_sekai)", "shida kazuhiro"),
    (2342188, 457, 664, "nikaidou_shinku", "shida kazuhiro"),
    (2342188, 665, 764, "toumine_tsukasa", "gt"),
    
    (836513, 1, 296, "nikaidou_shinku", "shida kazuhiro"),

    (2342201, 1, 263, "nikaidou_shinku", "shida kazuhiro"),
    # stralAir no Shiroki Towa
    (722276, 2, 156, "yuunagi_ichika", "shida kazuhiro"),
    (722276, 157, 275, "korona_(astralair)", "shida kazuhiro"),
    (722276, 276, 415, "mizunose_kotori", "shida kazuhiro"),
    (722276, 416, 563, "tachibana_ochiba", "shida kazuhiro"),
    (722276, 572, 686, "hotaru_rinne", "shida kazuhiro"),
    (722276, 687, 836, "yuuki_(astralair_no_shiroki_towa)", "shida kazuhiro"),

    (3525339, 79, 234, "yuunagi_ichika", "shida kazuhiro"),
    (3525339, 235, 353, "korona_(astralair)", "shida kazuhiro"),
    (3525339, 354, 491, "mizunose_kotori", "shida kazuhiro"),
    (3525339, 492, 638, "tachibana_ochiba", "shida kazuhiro"),
    (3525339, 647, 761, "hotaru_rinne", "shida kazuhiro"),
    (3525339, 762, 914, "yuuki_(astralair_no_shiroki_towa)", "shida kazuhiro"),
    (3525339, 975, 1033, "yuunagi_ichika", "shida kazuhiro"),
    (3525339, 1034, 1074, "korona_(astralair)", "shida kazuhiro"),
    (3525339, 1075, 1126, "mizunose_kotori", "shida kazuhiro"),
    (3525339, 1127, 1166, "tachibana_ochiba", "shida kazuhiro"),
    (3525339, 1167, 1211, "hotaru_rinne", "shida kazuhiro"),
    (3525339, 1212, 1334, "shiraha_yuki", "shida kazuhiro"),
    (3525339, 1335, 1378, "yuuki_(astralair_no_shiroki_towa)", "shida kazuhiro"),

    (759347, 150, 321, "tachibana_ochiba", "shida kazuhiro"),
    (759347, 322, 439, "yuunagi_ichika", "shida kazuhiro"),
    (759347, 537, 680, "hotaru_rinne", "shida kazuhiro"),
    (759347, 861, 964, "mizunose_kotori", "shida kazuhiro"),
    (759347, 965, 1106, "yuuki_(astralair_no_shiroki_towa)", "shida kazuhiro"),
    (759347, 1131, 1237, "korona_(astralair)", "shida kazuhiro"),

    (1023004, 91, 168, "yuunagi_ichika", "shida kazuhiro"),
    (1023004, 169, 237, "korona_(astralair)", "shida kazuhiro"),
    (1023004, 238, 324, "mizunose_kotori", "shida kazuhiro"),
    (1023004, 325, 368, "tachibana_ochiba", "shida kazuhiro"),
    (1023004, 378, 447, "hotaru_rinne", "shida kazuhiro"),
    (1023004, 520, 678, "shiraha_yuki", "shida kazuhiro"),
    (1023004, 679, 774, "yuuki_(astralair_no_shiroki_towa)", "shida kazuhiro"),

    # Hoshizora no Memoria
    (876019, 4, 119, "minahoshi_asuho", "shida kazuhiro"),
    (876019, 134, 217, "isuzu_aoi", "shida kazuhiro"),
    (876019, 218, 344, "hisakaki_komomo", "shida kazuhiro"),
    (876019, 346, 413, "hisakaki_kosame", "shida kazuhiro"),
    (876019, 414, 500, "mare_s._ephemeral", "shida kazuhiro"),
    (876019, 501, 584, "kogasaka_chinami", "shida kazuhiro"),
    (876019, 637, 658, "ototsu_yume", "shida kazuhiro"),

    (1536534, 4, 119, "minahoshi_asuho", "shida kazuhiro"),
    (1536534, 120, 203, "isuzu_aoi", "shida kazuhiro"),
    (1536534, 204, 330, "hisakaki_komomo", "shida kazuhiro"),
    (1536534, 332, 399, "hisakaki_kosame", "shida kazuhiro"),
    (1536534, 400, 486, "mare_s._ephemeral", "shida kazuhiro"),
    (1536534, 487, 570, "kogasaka_chinami", "shida kazuhiro"),
    (1536534, 635, 646, "ototsu_yume", "shida kazuhiro"),

    (878532, 5, 27, "minahoshi_asuho", "shida kazuhiro"),
    (878532, 102, 114, "hisakaki_komomo", "shida kazuhiro"),
    (878532, 115, 134, "hisakaki_kosame", "shida kazuhiro"),
    (878532, 135, 253, "mare_s._ephemeral", "shida kazuhiro"),
    (878532, 254, 270, "kogasaka_chinami", "shida kazuhiro"),
    (878532, 271, 394, "ototsu_yume", "shida kazuhiro"),

    (1536535, 4, 26, "minahoshi_asuho", "shida kazuhiro"),
    (1536535, 27, 41, "isuzu_aoi", "shida kazuhiro"),
    (1536535, 42, 74, "hisakaki_komomo", "shida kazuhiro"),
    (1536535, 75, 94, "hisakaki_kosame", "shida kazuhiro"),
    (1536535, 95, 213, "mare_s._ephemeral", "shida kazuhiro"),
    (1536535, 214, 230, "kogasaka_chinami", "shida kazuhiro"),
    (1536535, 231, 357, "ototsu_yume", "shida kazuhiro"),

    # Sakura, Moyu
    (1357385, 32, 185, "andou_chiwa", "natsume eri"),
    (1357385, 186, 432, "hiiragi_haru", "natsume eri"),
    (1357385, 433, 568, "yorutsuki_hiori", "natsume eri"),
    (1357385, 569, 779, "kuro_(sakura_moyu)", "shida kazuhiro"),

    (2693927, 29, 182, "andou_chiwa", "natsume eri"),
    (2693927, 277, 523, "hiiragi_haru", "natsume eri"),
    (2693927, 524, 659, "yorutsuki_hiori", "natsume eri"),
    (2693927, 660, 870, "kuro_(sakura_moyu)", "shida kazuhiro"),

    # Happy Live Show Up!
    (1921952, 4, 323, "sophia_turina", "bekotarou"),
    (1921952, 324, 526, "karentia_veribel", "bekotarou"),
    (1921952, 527, 709, "ru_mao", "bekotarou"),
    (1921952, 710, 867, "clarice_clawnya", "bekotarou"),
    (1921952, 868, 1123, "pechka_monica", "bekotarou"),
    (1921952, 1124, 1156, "miyabi_asahina", "bekotarou"),
    (1921952, 1160, 1235, "miyabi_asahina", "bekotarou"),

    (2595581, 2, 51, "clarice_clawnya", "bekotarou"),
    (2595581, 66, 137, "karentia_veribel", "bekotarou"),
    (2595581, 138, 198, "miyabi_asahina", "bekotarou"),
    (2595581, 244, 258, "miyabi_asahina", "bekotarou"),
    (2595581, 317, 427, "miyabi_asahina", "bekotarou"),
    (2595581, 505, 552, "ru_mao", "bekotarou"),

    (2595656, 5, 54, "clarice_clawnya", "bekotarou"),
    (2595656, 69, 140, "karentia_veribel", "bekotarou"),
    (2595656, 141, 203, "miyabi_asahina", "bekotarou"),
    (2595656, 249, 263, "miyabi_asahina", "bekotarou"),
    (2595656, 322, 432, "miyabi_asahina", "bekotarou"),
    (2595656, 499, 558, "ru_mao", "bekotarou"),
    # Natsuiro Koi Uta
    (998651, 3, 51, "tsukiyono_yuna", "lucie"),
    (998651, 56, 87, "takanashi_hikaru", "great mosu"),
    (998651, 95, 137, "kamiki_ayaka", "lucie"),
    (998651, 139, 162, "suwa_shion", "great mosu"),
    (998651, 170, 495, "tsukiyono_yuna", "lucie"),
    (998651, 496, 783, "takanashi_hikaru", "great mosu"),
    (998651, 784, 1081, "kamiki_ayaka", "lucie"),
    (998651, 1082, 1434, "suwa_shion", "great mosu"),

    (1228859, 1, 360, "kamiki_ayaka", "lucie"),
    (1228859, 361, 720, "takanashi_hikaru", "great mosu"),
    (1228859, 731, 915, "suwa_shion", "great mosu"),
    (1228859, 916, 1385, "tsukiyono_yuna", "lucie"),
    (1228859, 1386, 1745, "kamiki_ayaka", "lucie"),
    (1228859, 1746, 2000, "takanashi_hikaru", "great mosu"),

    (1228863, 1, 105, "takanashi_hikaru", "great mosu"),
    (1228863, 116, 300, "suwa_shion", "great mosu"),
    (1228863, 301, 770, "tsukiyono_yuna", "lucie"),

    # Kimi to Tsunagaru Koi Flag
    (1230384, 4, 324, None, "lucie"),
    (1230384, 325, 667, None, "lucie"),
    (1230384, 668, 1113, "amano_misaki_(koi_flag) ", "kakao"),
    (1230384, 1114, 1494, None, "lucie"),
    # Karenai Sekai to Owaru Hana
    (1000105, 3, 114, "haru_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (1000105, 115, 231, "kotose_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (1000105, 232, 275, "ren_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (1000105, 287, 401, "yukina_(karenai_sekai_to_owaru_hana)", "ameto yuki"),

    (2289746, 55, 398, "kotose_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (2289746, 434, 847, "haru_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (2289746, 881, 1243, "yukina_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    (2289746, 1244, 1618, "ren_(karenai_sekai_to_owaru_hana)", "ameto yuki"),
    # Kemono Musume no Sodatekata
    (1293763, 4, 10, "sakuraba_iroha", "motomiya mitsuki"),
    (1293763, 11, 15, "kagaya_kana", "motomiya mitsuki"),
    (1293763, 16, 30, "sakuraba_iroha", "motomiya mitsuki"),
    (1293763, 31, 36, "kagaya_kana", "motomiya mitsuki"),
    (1293763, 37, 78, "sakuraba_iroha", "motomiya mitsuki"),
    (1293763, 99, 114, "kagaya_kana", "motomiya mitsuki"),
    (1293763, 115, 136, "sakuraba_iroha", "motomiya mitsuki"),
    (1293763, 137, 151, "kagaya_kana", "motomiya mitsuki"),
    (1293763, 152, 167, "sakuraba_iroha", "motomiya mitsuki"),
    (1293763, 168, 189, "sakuraba_iroha, kagaya_kana", "motomiya mitsuki"),

    (1499756, 7, 12, "sakuraba_iroha", "motomiya mitsuki"),
    (1499756, 13, 17, "kagaya_kana", "motomiya mitsuki"),
    (1499756, 18, 32, "sakuraba_iroha", "motomiya mitsuki"),
    (1499756, 33, 38, "kagaya_kana", "motomiya mitsuki"),
    (1499756, 39, 62, "sakuraba_iroha", "motomiya mitsuki"),
    (1499756, 101, 116, "kagaya_kana", "motomiya mitsuki"),
    (1499756, 117, 138, "sakuraba_iroha", "motomiya mitsuki"),
    (1499756, 139, 153, "kagaya_kana", "motomiya mitsuki"),
    (1499756, 154, 169, "sakuraba_iroha", "motomiya mitsuki"),
    (1499756, 170, 191, "sakuraba_iroha, kagaya_kana", "motomiya mitsuki"),

    (3656996, 1, 1434, "sakuraba_iroha", "motomiya mitsuki"),
    (3656996, 1642, 1901, "kagaya_kana", "motomiya mitsuki"),
    (3656996, 1, 2000, None, "motomiya mitsuki"),

    # Nekogami-sama to, Nanatsuboshi -Imouto no Ane-
    (1369335, 10, 91, "aoba_erika", "suimya"),
    (1369335, 95, 107, "nanahara_fuuko", "suimya"),
    (3656994, 1, 960, "aoba_erika", "suimya"),
    (3656995, 1, 713, "nanahara_fuuko", "suimya"),
    # Asagao wa Ai o Shiranai.
    (1972589, 1, 235, "hiiro_asagao", "maumen"),
    (1972589, 240, 381, "karasuba_kotori", "maumen"),
    (1972589, 382, 532, "kamishiro_miko", "maumen"),
    # 2045, Tsuki yori. - The Butterfly Dream
    (2339354, 74, 363, "el_(2045, tsuki yori)", "maumen"),
    (2339354, 549, 796, "hina_yui", "maumen"),
    (2339354, 858, 1194, "sakuratsuki_shirabe", "maumen"),
    # Sky Chord
    (2936456, 266, 489, "tenshi_(sky_chord)", "maumen"),
    (2936456, 509, 708, "urushiduki_shizuku", "maumen"),
    # True Colors
    (3704896, 66, 222, "natsuake_hotaru", "maumen"),
    (3704896, 261, 482, "iori_minamo", "maumen"),
    (3704896, 528, 549, "miseki_soubi", "maumen"),
    (3704896, 551, 614, "miseki_soubi", "maumen"),
    # Kirakira Stars
    # (1572826, 2, 95, "amane_ai", None),
    # (1678132, 1, 84, "yukishiro_nagisa", None),
    # (1678132, 87, 180, "yukishiro_nagisa", None),
    # (1766085, 1, 267, "aiba_reika", None),
    # Kanzume Shoujo no Shuumatsu Sekai
    (1403984, 120, 206, "sarasa_sari", "ichiri"),
    (1403984, 207, 241, "yaotome_hanae", "ichiri"),
    (1403984, 242, 263, "tsujika_saki", "ichiri"),
    (1403984, 290, 380, "sarasa_sari", "ichiri"),
    (1403984, 381, 461, "yaotome_hanae", "ichiri"),
    (1403984, 462, 525, "tsujika_saki", "ichiri"),
    (1403984, 526, 589, "yaotome_hanae", "ichiri"),
    (1403984, 930, 944, "tsujika_saki", "ichiri"),
    (1403984, 1005, 1067, "tsujika_saki", "ichiri"),
    (1403984, 1110, 1186, "sarasa_sari", "ichiri"),

    (1413770, 50, 169, "sarasa_sari", "ichiri"),
    (1413770, 292, 411, "yaotome_hanae", "ichiri"),
    (1413770, 170, 291, "tsujika_saki", "ichiri"),

    (1978045, 1, 2000, "yaotome_hanae", "ichiri"),
    # Koi x Mitsu 
    (1900094, 9, 162, "yaeneri_saki", "ichiri"),
    (1900094, 169, 269, "yaeneri_saki", "ichiri"),
    (1900095, 11, 164, "yaeneri_saki", "ichiri"),
    (1900095, 182, 285, "yaeneri_saki", "ichiri"),
    # Ninja Girl and the Mysterious Army of Urban Legend Monsters!
    (1437787, 1, 2000, "hattori_kiri", "matsumiya kiseri"),
    (1437819, 1, 1894, "hattori_kiri", "matsumiya kiseri"),
    (1437819, 1895, 2000, "lily_(ninja_girl)", "picpicgram"),
    (1437820, 1, 2000, "lily_(ninja_girl)", "picpicgram"),
    (1441286, 1, 160, "lily_(ninja_girl)", "picpicgram"),
    (1441286, 161, 752, "mary_(ninja_girl)", "matsumiya kiseri"),
    (1441286, 753, 1544, "hattori_kiri", "matsumiya kiseri"),
    (1441286, 1545, 2000, "lily_(ninja_girl)", "picpicgram"),
    (1441284, 1, 258, "lily_(ninja_girl)", "picpicgram"),
    # Natsu no Iro no Nostalgia
    (937359, 111, 117, "manazuru_misaki", "yamakaze ran"),
    (937359, 119, 138, "manazuru_misaki", "yamakaze ran"),
    (937359, 141, 341, "manazuru_misaki", "yamakaze ran"),
    (937359, 342, 519, "shinjouji_ayane", "yamakaze ran"),
    (937359, 520, 725, "orikuchi_miu", "yamakaze ran"),
    (937359, 726, 947, "maniwa_shouko", "yamakaze ran"),

    (1761687, 11, 835, "manazuru_misaki", "yamakaze ran"),
    (1761687, 836, 1681, "shinjouji_ayane", "yamakaze ran"),
    (1761687, 1682, 2000, "orikuchi_miu", "yamakaze ran"),
    (1761694, 1, 605, "orikuchi_miu", "yamakaze ran"),
    (1761694, 606, 1585, "maniwa_shouko", "yamakaze ran"),
    # Sakura no Mori Dreamers
    (938915, 3, 183, "kiritou_kureha", "yamakaze ran"),
    (938915, 184, 367, "shizumiya_mahoro", "yamakaze ran"),
    (938915, 368, 549, "erisaka_mifuyu", "yamakaze ran"),
    (938915, 550, 751, "fukigami_hatsune", "yamakaze ran"),

    (939312, 1, 136, "kiritou_kureha", "yamakaze ran"),
    (939312, 137, 217, "shizumiya_mahoro", "yamakaze ran"),
    (939312, 218, 313, "erisaka_mifuyu", "yamakaze ran"),
    (939312, 314, 393, "fukigami_hatsune", "yamakaze ran"),
    (939312, 443, 489, "akitsu_madoka", "yamakaze ran"),

    (1092850, 493, 691, "akitsu_madoka", "yamakaze ran"),
    (1092850, 726, 761, "akitsu_madoka", "yamakaze ran"),

    (1446339, 1, 2000, "kiritou_kureha", "yamakaze ran"),

    (1446354, 1, 969, "shizumiya_mahoro", "yamakaze ran"),
    (1446354, 970, 2000, "erisaka_mifuyu", "yamakaze ran"),

    (1446370, 1, 499, "erisaka_mifuyu", "yamakaze ran"),
    (1446370, 500, 2000, "fukigami_hatsune", "yamakaze ran"),

    (1446382, 1, 792, "fukigami_hatsune", "yamakaze ran"),
    (1446382, 793, 2000, "akitsu_madoka", "yamakaze ran"),

    (1540381, 1, 246, "kiritou_kureha", "yamakaze ran"),
    (1540381, 247, 504, "erisaka_mifuyu", "yamakaze ran"),
    (1540381, 505, 1440, "akitsu_madoka", "yamakaze ran"),
    (1540461, 1, 468, "akitsu_madoka", "yamakaze ran"),

    (1610176, 301, 497, "kiritou_kureha", "yamakaze ran"),
    (1610176, 498, 687, "shizumiya_mahoro", "yamakaze ran"),
    (1610176, 688, 869, "erisaka_mifuyu", "yamakaze ran"),
    (1610176, 870, 1072, "fukigami_hatsune", "yamakaze ran"),
    # Uchi no __
    (1244947, 1, 294, "matoi_ayame", "yamakaze ran"),
    (1993949, 1, 293, "matoi_ayame", "yamakaze ran"),

    (1321831, 1, 292, "tachibanaki_riho", "yamakaze ran"),
    (2206673, 1, 288, "tachibanaki_riho", "yamakaze ran"),

    (1403915, 82, 102, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 103, 122, "chika_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 123, 136, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 137, 160, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 161, 174, "chika_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 175, 188, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 189, 194, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 195, 202, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1403915, 203, 218, "chika_(uchi_no_koibito)", "yamakaze ran"),

    (2563667, 68, 88, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 89, 108, "chika_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 109, 122, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 123, 146, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 147, 160, "chika_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 161, 174, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 175, 180, "chika_(uchi_no_koibito), yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 181, 188, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (2563667, 189, 204, "chika_(uchi_no_koibito)", "yamakaze ran"),

    (1880043, 1, 9, "matoi_ayame", "yamakaze ran"),

    (1880043, 10, 2000, "matoi_ayame", "yamakaze ran"),
    (1880055, 1, 74, "matoi_ayame", "yamakaze ran"),
    (1880055, 75, 2000, "tachibanaki_riho", "yamakaze ran"),
    (1880077, 1, 650, "tachibanaki_riho", "yamakaze ran"),
    (1880077, 651, 1946, "chika_(uchi_no_koibito)", "yamakaze ran"),
    (1880077, 1947, 2000, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    (1880098, 1, 1278, "yuri_(uchi_no_koibito)", "yamakaze ran"),
    # Bishoujo Gakuen -
    (1621415, 1, 368, "amamiya_meiko", "yamakaze ran"),
    (1621415, 372, 375, "amamiya_meiko", "yamakaze ran"),
    (1621415, 396, 403, "amamiya_meiko", "yamakaze ran"),

    (1740921, 1, 5, "takatori_shizuru", "yamakaze ran"),
    (1740921, 17, 290, "takatori_shizuru", "yamakaze ran"),
    (1740921, 322, 332, "takatori_shizuru", "yamakaze ran"),

    (1741734, 59, 332, "takatori_shizuru", "yamakaze ran"),
    (1741734, 364, 374, "takatori_shizuru", "yamakaze ran"),
    (1741734, 401, 524, "takatori_shizuru", "yamakaze ran"),
    (1741734, 526, 584, "misono_emiri", "yamakaze ran"),

    (1876084, 14, 384, "misono_emiri", "yamakaze ran"),
    (1876084, 435, 439, "misono_emiri", "yamakaze ran"),

    (1878710, 5, 844, "amamiya_meiko", "yamakaze ran"),
    (1878710, 845, 2000, "takatori_shizuru", "yamakaze ran"),
    (1878719, 1, 1916, "takatori_shizuru", "yamakaze ran"),
    (1878719, 1917, 2000, "misono_emiri", "yamakaze ran"),
    (1878725, 1, 1196, "misono_emiri", "yamakaze ran"),
    # __ wa Iyashite Agetai
    (2206722, 1, 601, "mamiya_mami", "yamakaze ran"),

    (2339665, 37, 60, "kouzu_yuuri, takanashi_marin", "yamakaze ran"),
    (2339665, 61, 81, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 82, 148, "kouzu_yuuri, takanashi_marin", "yamakaze ran"),
    (2339665, 149, 201, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 202, 230, "takanashi_marin", "yamakaze ran"),
    (2339665, 231, 247, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 248, 270, "takanashi_marin", "yamakaze ran"),
    (2339665, 271, 284, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 285, 305, "takanashi_marin", "yamakaze ran"),
    (2339665, 306, 332, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 333, 355, "takanashi_marin", "yamakaze ran"),
    (2339665, 356, 401, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 402, 423, "takanashi_marin", "yamakaze ran"),
    (2339665, 424, 476, "kouzu_yuuri, takanashi_marin", "yamakaze ran"),
    (2339665, 477, 495, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 496, 517, "takanashi_marin", "yamakaze ran"),
    (2339665, 518, 550, "kouzu_yuuri, takanashi_marin", "yamakaze ran"),
    (2339665, 551, 564, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 565, 567, "kouzu_yuuri, takanashi_marin", "yamakaze ran"),
    (2339665, 568, 569, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 608, 620, "mamiya_mami", "yamakaze ran"),
    (2339665, 621, 645, "kouzu_yuuri", "yamakaze ran"),
    (2339665, 646, 695, "takanashi_marin", "yamakaze ran"),
    (2339665, 696, 704, "oshino_ruka", "yamakaze ran"),

    (2479920, 41, 376, "oshino_ruka", "yamakaze ran"),
    (2479920, 534, 578, "mamiya_mami", "yamakaze ran"),
    (2479920, 579, 628, "kouzu_yuuri", "yamakaze ran"),
    (2479920, 629, 728, "takanashi_marin", "yamakaze ran"),
    (2479920, 729, 786, "oshino_ruka", "yamakaze ran"),
    # Natsu e no Hakobune
    (2935526, 1, 410, "takahagi_ruriko", "yamakaze ran"),
    (2935526, 416, 442, "takahagi_ruriko", "yamakaze ran"),

    (3070802, 8, 371, "takahagi_miku", "yamakaze ran"),

    (3216426, 7, 335, "kinusa_shiori", "yamakaze ran"),
    (3216426, 348, 363, "takahagi_ruriko", "yamakaze ran"),
    (3216426, 364, 378, "takahagi_miku", "yamakaze ran"),
    (3216426, 379, 387, "kinusa_shiori", "yamakaze ran"),
    # Sex Open World
    (1491661, 5, 323, "kashima_hikari", "hinata nao"),
    (1491661, 324, 652, "alencia_yonah_elborg", "hinata nao"),
    (1491661, 653, 990, "ehlitte", "hinata nao"),
    (1491661, 991, 1289, "madri_parnepoli", "hinata nao"),
    (1491661, 1290, 1581, "makibashira_yuuna", "hinata nao"),
    (1491661, 1582, 1912, "anna_heiritte", "hinata nao"),
    (1491674, 1, 312, "elda_heiritte", "hinata nao"),

    (1492739, 1, 150, "kashima_hikari", "hinata nao"),
    (1492739, 151, 320, "alencia_yonah_elborg", "hinata nao"),
    (1492739, 321, 465, "ehlitte", "hinata nao"),
    (1492739, 466, 565, "madri_parnepoli", "hinata nao"),
    (1492739, 566, 673, "makibashira_yuuna", "hinata nao"),
    (1492739, 674, 818, "anna_heiritte", "hinata nao"),
    (1492739, 819, 974, "elda_heiritte", "hinata nao"),

    (3173960, 2, 99, "kashima_hikari", "hinata nao"),
    (3173960, 100, 199, "alencia_yonah_elborg", "hinata nao"),
    (3173960, 200, 275, "ehlitte", "hinata nao"),
    (3173960, 276, 344, "madri_parnepoli", "hinata nao"),
    (3173960, 345, 422, "makibashira_yuuna", "hinata nao"),
    (3173960, 423, 506, "anna_heiritte", "hinata nao"),
    (3173960, 507, 594, "elda_heiritte", "hinata nao"),
    # Imouto Paradise! 3
    (1176575, 3, 296, "nanase_sakura", "hinata nao"),
    (1176575, 297, 592, "nanase_rika", "hinata nao"),
    (1176575, 593, 878, "nanase_hinata", "hinata nao"),
    (1176575, 879, 1174, "nanase_hiharu", "hinata nao"),
    (1176575, 1175, 1476, "nanase_zakuro", "hinata nao"),

    (1188694, 4, 297, "nanase_sakura", "hinata nao"),
    (1188694, 298, 593, "nanase_rika", "hinata nao"),
    (1188694, 594, 879, "nanase_hinata", "hinata nao"),
    (1188694, 880, 1175, "nanase_hiharu", "hinata nao"),
    (1188694, 1176, 1477, "nanase_zakuro", "hinata nao"),

    (1378994, 1, 1976, "nanase_sakura", "hinata nao"),
    (1379002, 1, 304, "nanase_sakura", "hinata nao"),
    (1379002, 305, 1976, "nanase_hiharu", "hinata nao"),
    (1379032, 1, 380, "nanase_hiharu", "hinata nao"),
    (1379032, 381, 1824, "nanase_hinata", "hinata nao"),
    (1379058, 1, 608, "nanase_hinata", "hinata nao"),
    (1379058, 609, 1976, "nanase_rika", "hinata nao"),
    (1379070, 1, 518, "nanase_rika", "hinata nao"),
    (1379070, 519, 1976, "nanase_zakuro", "hinata nao"),
    (1379136, 1, 740, "nanase_zakuro", "hinata nao"),

    (1571576, 61, 122, "nanase_sakura", "hinata nao"),
    (1571576, 123, 201, "nanase_rika", "hinata nao"),
    (1571576, 202, 264, "nanase_hinata", "hinata nao"),
    (1571576, 265, 340, "nanase_hiharu", "hinata nao"),
    (1571576, 341, 414, "nanase_zakuro", "hinata nao"),

    (1571778, 4, 297, "nanase_sakura", "hinata nao"),
    (1571778, 298, 593, "nanase_rika", "hinata nao"),
    (1571778, 594, 879, "nanase_hinata", "hinata nao"),
    (1571778, 880, 1175, "nanase_hiharu", "hinata nao"),
    (1571778, 1176, 1477, "nanase_zakuro", "hinata nao"),
    # Yuunagi-sou no S-kyuu no Kanojo-tachi
    (2361983, 4, 288, "shishido_manaka", "hinata nao"),
    (2361983, 292, 556, "taira_misaki", "hinata nao"),
    (2361983, 559, 898, "kurumadani_non", "hinata nao"),
    (2361983, 902, 1148, "ninagawa_sakura", "hinata nao"),

    (2389088, 1, 87, "shishido_manaka", "hinata nao"),
    (2389088, 88, 237, "taira_misaki", "hinata nao"),
    (2389088, 238, 337, "kurumadani_non", "hinata nao"),
    (2389088, 338, 442, "ninagawa_sakura", "hinata nao"),

    (2773433, 6, 447, "mikuriya_haruna", "hinata nao"),
    (2773433, 492, 850, "kuromine_ai", "hinata nao"),
    (2773433, 851, 1322, "takatsuki_taeko", "hinata nao"),
    (2773433, 1324, 1744, "ooigawa_risa", "hinata nao"),

    (2775456, 2, 121, "mikuriya_haruna", "hinata nao"),
    (2775456, 122, 208, "kuromine_ai", "hinata nao"),
    (2775456, 209, 324, "takatsuki_taeko", "hinata nao"),
    (2775456, 325, 504, "ooigawa_risa", "hinata nao"),
    # Amaekata wa Kanojo nari ni.
    (908674, 5, 77, "nagamine_kanae", "konomi"),
    (908674, 78, 161, "kurisawa_miyuki", "konomi"),
    (908674, 162, 163, "kurisawa_miyuki, shiga_nohana", "konomi"),
    (908674, 164, 222, "shiga_nohana", "konomi"),
    (908674, 223, 303, "niikura_tomomi", "konomi"),

    (1189542, 1, 1083, "nagamine_kanae", "konomi"),
    (1189542, 1423, 2000, "kurisawa_miyuki", "konomi"),
    (1189565, 1, 308, "kurisawa_miyuki", "konomi"),
    (1189565, 309, 1110, "shiga_nohana", "konomi"),
    (1189565, 1543, 2000, "niikura_tomomi", "konomi"),
    (1189576, 1, 201, "niikura_tomomi", "konomi"),
    # Shirogane × Spirits!
    (803571, 38, 95, "kiryuu_chikage", "konomi"),
    (803571, 148, 225, "kamura_masaki", "konomi"),
    (803571, 226, 312, "nakano_mei", "konomi"),
    (803571, 313, 385, "shinonome_setsuna", "konomi"),
    (803571, 386, 407, "kiryuu_chikage", "konomi"),
    (803571, 416, 421, "kamura_masaki", "konomi"),
    (803571, 422, 431, "nakano_mei", "konomi"),
    (803571, 432, 438, "shinonome_setsuna", "konomi"),
    # Natural Vacation
    (1218921, 3, 243, "fujisaki_haruka", "asami asami"),
    (1218921, 244, 446, "sarashina_yuzuki", "asami asami"),

    (2261821, 2, 230, "fujisaki_haruka", "asami asami"),
    (2261821, 231, 432, "sarashina_yuzuki", "asami asami"),
    # Koi wa Sotto Saku Hana no You ni
    (1245620, 19, 92, "kotoishi_iori", "gin"),
    (1245620, 93, 185, "hayami_misato", "keisaka"),
    (1245620, 186, 268, "miyane_saki", "sena chifuyu"),
    (1245620, 269, 365, "kurumi_sanae", "gin"),
    (1245620, 366, 462, "saionji_youko", "kotarou"),
    (1245620, 463, 497, "toudou_nazuna", "keisaka"),

    (1289430, 1, 240, "saionji_youko", "kotarou"),
    (1289430, 265, 520, "hayami_misato", "keisaka"),
    (1289430, 521, 760, "kotoishi_iori", "gin"),
    (1289430, 761, 984, "kurumi_sanae", "gin"),
    (1289430, 985, 1176, "miyane_saki", "sena chifuyu"),
    (1289430, 1226, 1449, "toudou_nazuna", "keisaka"),

    (1389151, 11, 62, "kotoishi_iori", "gin"),
    (1389151, 63, 104, "hayami_misato", "keisaka"),
    (1389151, 108, 160, "toudou_nazuna", "keisaka"),
    (1389151, 161, 218, "miyane_saki", "sena chifuyu"),
    (1389151, 219, 268, "kurumi_sanae", "gin"),
    (1389151, 269, 328, "saionji_youko", "kotarou"),

    (1443967, 1, 768, "saionji_youko", "kotarou"),
    (1443967, 769, 1440, "kotoishi_iori", "gin"),
    (1443967, 1441, 2000, "hayami_misato", "keisaka"),
    (1443989, 1, 192, "hayami_misato", "keisaka"),
    (1443989, 193, 752, "miyane_saki", "sena chifuyu"),
    (1443989, 753, 1392, "kurumi_sanae", "gin"),
    (1443989, 1393, 1840, "toudou_nazuna", "keisaka"),
    # Hana wa Mijikashi, Odoreyo Otome
    (2344212, 43, 263, "christina_white", "mizuki yuuma"),
    (2344212, 264, 508, "melissa_leoni", "teeta.j"),
    (2344212, 509, 726, "tsukubane_shinobu", "nanotaro"),
    (2344212, 735, 909, "yuzuriha_koharu", "mizuki yuuma"),

    (3457526, 216, 435, "christina_white", "mizuki yuuma"),
    (3457526, 436, 645, "melissa_leoni", "teeta.j"),
    (3457526, 646, 817, "tsukubane_shinobu", "nanotaro"),
    (3457526, 818, 989, "yuzuriha_koharu", "mizuki yuuma"),

    (3457527, 216, 435, "christina_white", "mizuki yuuma"),
    (3457527, 436, 645, "melissa_leoni", "teeta.j"),
    (3457527, 646, 817, "tsukubane_shinobu", "nanotaro"),
    (3457527, 818, 989, "yuzuriha_koharu", "mizuki yuuma"),
    # Natsu no Majo no Parade
    (981373, 57, 222, "alisa_crowley", "annie"),
    (981373, 223, 348, "yugamo_azuki", "annie"),
    (981373, 349, 498, "carol_mercurius", "annie"),
    (981373, 499, 690, "amatsu_sasha", "annie"),

    (1135217, 1, 196, "alisa_crowley", "annie"),
    (1135217, 197, 518, "carol_mercurius", "annie"),
    (1135217, 519, 753, "yugamo_azuki", "annie"),
    (1135218, 1, 94, "yugamo_azuki", "annie"),
    (1135218, 95, 571, "amatsu_sasha", "annie"),
    # Fuun to Kouun to Koiuranai no Tarot
    (1026937, 1, 2000, "hoshimi_akane", "annie"),
    # Aikotoba -Silver Snow Sister
    (2718658, 84, 470, "hoshitsugu_shirone", "annie"),
    (2718658, 528, 766, "hoshitsugu_shirone", "annie"),
    (3418294, 1, 2000, "hoshitsugu_shirone", "annie"),
    (3418295, 1, 2000, "hoshitsugu_shirone", "annie"),
    # Rhapsodic Holiday
    (1246523, 11, 29, "hanyuuin_yuzuha", "annie"),
    (1246523, 30, 53, "maha_d_bancroft", "hinata mutsuki"),
    (1246523, 54, 72, "inaishi_arisa", "sasahiro"),
    (1246523, 73, 88, "aeba_natsuki", "annie"),
    (1246523, 95, 168, "hanyuuin_yuzuha", "annie"),
    (1246523, 169, 243, "maha_d_bancroft", "hinata mutsuki"),
    (1246523, 244, 318, "inaishi_arisa", "sasahiro"),
    (1246523, 319, 393, "aeba_natsuki", "annie"),
    # Makimura Hazuki no Koigatari
    (2020008, 64, 400, "makimura_hazuki", "annie"),
    (2020008, 426, 500, "makimura_hazuki", "annie"),

    (3443492, 31, 630, "makimura_hazuki", "annie"),
    (3443492, 669, 1519, "makimura_hazuki", "annie"),

    (3443494, 31, 630, "makimura_hazuki", "annie"),
    (3443494, 669, 1519, "makimura_hazuki", "annie"),
    # Lovesick Puppies
    (577257, 66, 190, "karakouji_orie", "sankuro"),
    (577257, 191, 287, "shibasaki_maruna", "sankuro"),
    (577257, 295, 429, "himesato_isami", "sankuro"),
    (577257, 430, 531, "sofiya_alecseevna_feofanova", "sankuro"),
    (577257, 532, 640, "hoshina_yuki", "sankuro"),

    (748219, 2, 144, "karakouji_orie", "sankuro"),
    (748219, 145, 300, "shibasaki_maruna", "sankuro"),
    (748219, 301, 453, "sofiya_alecseevna_feofanova", "sankuro"),
    (748219, 454, 612, "himesato_isami", "sankuro"),
    (748219, 613, 768, "hoshina_yuki", "sankuro"),
    # HGG Megami no Shuuen
    (651489, 1, 813, None, "miyasu risa"),
    (651489, 814, 2000, None, "nanase meruchi"),
    (651490, 1, 451, None, "nanase meruchi"),
    (651490, 452, 811, None, "miyasu risa"),
    (651490, 812, 2000, None, "nanase meruchi"),
    # RepKiss
    (917775, 8, 203, "itsugaya_hayane", "mikoto akemi"),
    (917775, 204, 359, "itsugaya_kanade", "mikoto akemi"),
    (917775, 360, 585, "futaba_saki", "hitsuji takako"),
    (917775, 601, 843, "futaba_yuiri", "unasaka"),
    (917775, 844, 847, "itsugaya_hayane", "mikoto akemi"),
    (917775, 848, 855, "itsugaya_kanade", "mikoto akemi"),
    (917775, 856, 864, "futaba_saki", "hitsuji takako"),
    (917775, 865, 870, "futaba_yuiri", "unasaka"),

    (3426954, 1, 753, "itsugaya_hayane", "mikoto akemi"),
    (3426954, 754, 1236, "itsugaya_kanade", "mikoto akemi"),
    (3426954, 1237, 1723, "futaba_saki", "hitsuji takako"),
    (3426955, 1, 512, "futaba_yuiri", "unasaka"),
    # kiss art
    (670561, 50, 296, "aikawa_arisa", "mikoto akemi"),
    (670561, 297, 511, "natsume_azusa", "hitsuji takako"),
    (670561, 512, 767, "minase_yuka", "takei ooki"),
    (670561, 768, 1000, "hoshimi_tsukuyo", "mikoto akemi"),

    (3428368, 1, 296, "aikawa_arisa", "mikoto akemi"),
    (3428368, 297, 568, "natsume_azusa", "hitsuji takako"),
    (3428368, 681, 1010, "minase_yuka", "takei ooki"),
    (3428368, 1011, 1696, "hoshimi_tsukuyo", "mikoto akemi"),
    # Ai Kiss
    (1508822, 77, 204, "saegusa_ayame", "kirisawa saki"),
    (1508822, 205, 321, "saegusa_hinata", "kirisawa saki"),
    (1508822, 322, 444, "kanno_junko", "haduki gyokuto"),
    (1508822, 445, 575, "sakurada_an", "haduki gyokuto"),

    (1520666, 4, 146, "saegusa_ayame", "kirisawa saki"),
    (1520666, 147, 247, "saegusa_hinata", "kirisawa saki"),
    (1520666, 248, 370, "kanno_junko", "haduki gyokuto"),
    (1520666, 371, 501, "sakurada_an", "haduki gyokuto"),

    (1578594, 5, 149, "sakurada_an", "haduki gyokuto"),
    (1578594, 150, 299, "saegusa_ayame", "kirisawa saki"),
    (1578594, 375, 500, "saegusa_hinata", "kirisawa saki"),
    (1578594, 501, 632, "kanno_junko", "haduki gyokuto"),

    (1787632, 7, 111, "sakurada_an", "haduki gyokuto"),
    (1787632, 112, 142, "saegusa_ayame", "kirisawa saki"),
    (1787632, 221, 264, "saegusa_hinata", "kirisawa saki"),
    (1787632, 273, 350, "kanno_junko", "haduki gyokuto"),
    (1787632, 381, 495, "katsuragi_nanase", "kirisawa saki"),
    (1787632, 534, 654, "tosu_towako", "kirisawa saki"),

    (2190840, 2, 121, "sakurada_an", "haduki gyokuto"),
    (2190840, 122, 201, "saegusa_ayame", "kirisawa saki"),
    (2190840, 275, 324, "saegusa_hinata", "kirisawa saki"),
    (2190840, 325, 391, "kogure_ikue", "kirisawa saki"),
    (2190840, 392, 409, "kanno_junko", "haduki gyokuto"),
    (2190840, 410, 534, "maniwa_karin", "haduki gyokuto"),
    (2190840, 550, 634, "katsuragi_nanase", "kirisawa saki"),
    (2190840, 922, 945, "tosu_towako", "kirisawa saki"),

    (2449666, 5, 48, "katsuragi_nanase", "kirisawa saki"),

    (3428376, 1, 270, "sakurada_an", "haduki gyokuto"),
    (3428376, 271, 494, "saegusa_ayame", "kirisawa saki"),
    (3428376, 535, 738, "saegusa_hinata", "kirisawa saki"),
    (3428376, 739, 894, "kanno_junko", "haduki gyokuto"),

    (3428377, 1, 168, "sakurada_an", "haduki gyokuto"),
    (3428377, 169, 404, "saegusa_ayame", "kirisawa saki"),
    (3428377, 417, 914, "saegusa_hinata", "kirisawa saki"),
    (3428377, 915, 946, "kogure_ikue", "kirisawa saki"),
    (3428377, 947, 1076, "kanno_junko", "haduki gyokuto"),
    (3428377, 1109, 1221, "katsuragi_nanase", "kirisawa saki"),
    (3428377, 1284, 1393, "tosu_towako", "kirisawa saki"),

    (3428378, 1, 221, "sakurada_an", "haduki gyokuto"),
    (3428378, 222, 639, "saegusa_ayame", "kirisawa saki"),
    (3428378, 652, 1263, "saegusa_hinata", "kirisawa saki"),
    (3428378, 1264, 1448, "kogure_ikue", "kirisawa saki"),
    (3428379, 1, 130, "kanno_junko", "haduki gyokuto"),
    (3428379, 131, 409, "maniwa_karin", "haduki gyokuto"),
    (3428379, 442, 708, "katsuragi_nanase", "kirisawa saki"),
    (3428379, 1162, 1223, "tosu_towako", "kirisawa saki"),

    (3428382, 1, 67, "kogure_ikue", "kirisawa saki"),
    (3428382, 68, 149, "katsuragi_nanase", "kirisawa saki"),
    # Haru Kiss
    (803709, 17, 136, "hyoudou_amane", "mikoto akemi"),
    (803709, 137, 255, "shiraishi_aoi", "mikoto akemi"),
    (803709, 256, 343, "yasumi_itsuki", "marui"),
    (803709, 344, 433, "seto_konomi", "marui"),

    (810328, 1, 504, "shiraishi_aoi", "mikoto akemi"),
    (810328, 505, 864, "hyoudou_amane", "mikoto akemi"),
    (810328, 865, 1302, "yasumi_itsuki", "marui"),
    (810328, 1303, 1534, "seto_konomi", "marui"),

    (3428369, 1, 360, "hyoudou_amane", "mikoto akemi"),
    (3428369, 361, 864, "shiraishi_aoi", "mikoto akemi"),
    (3428369, 865, 1302, "yasumi_itsuki", "marui"),
    (3428369, 1341, 1630, "seto_konomi", "marui"),
    # Hotch Kiss
    (1006081, 4, 116, "ashikawa_yukino", "marui"),
    (1006081, 117, 218, "haruhino_misaki", "mikoto akemi"),
    (1006081, 219, 364, "sumiyoshi_nana", "marui"),
    (1006081, 365, 470, "mikage_shizuku", "mikoto akemi"),

    (3428359, 347, 742, "haruhino_misaki", "mikoto akemi"),
    (3428359, 756, 1142, "sumiyoshi_nana", "marui"),
    (3428359, 1143, 1471, "mikage_shizuku", "mikoto akemi"),
    (3428359, 1472, 1735, "ashikawa_yukino", "marui"),
    # Kiss Bell
    (1045784, 9, 188, "kajiya_ayano", "mikoto akemi"),
    (1045784, 189, 339, "takahata_chiharu", "mikoto akemi"),
    (1045784, 340, 532, "miyamae_eri", "takei ooki"),
    (1045784, 541, 632, "nagatsuda_yumi", "marui"),

    (1189853, 1, 1066, "kajiya_ayano", "mikoto akemi"),
    (1189853, 1067, 2000, "takahata_chiharu", "mikoto akemi"),
    (1189865, 1, 74, "takahata_chiharu", "mikoto akemi"),
    (1189865, 75, 1262, "miyamae_eri", "takei ooki"),
    (1189865, 1749, 2000, "nagatsuda_yumi", "marui"),
    (1189874, 1, 660, "nagatsuda_yumi", "marui"),
    # Full Kiss
    (1043680, 3, 113, "narahara_chisa", "mikoto akemi"),
    (1043680, 133, 225, "hosaki_sumire", "unasaka"),
    (1043680, 226, 331, "futakami_youko", "mikoto akemi"),
    (1043680, 332, 436, "yaegashi_yuuki", "unasaka"),
    (1043680, 437, 444, "narahara_chisa", "mikoto akemi"),
    (1043680, 445, 463, "hosaki_sumire", "unasaka"),
    (1043680, 464, 475, "futakami_youko", "mikoto akemi"),
    (1043680, 476, 485, "yaegashi_yuuki", "unasaka"),

    (1579344, 7, 103, "kawagiri_hana", "unasaka"),
    (1579344, 104, 216, "shinjou_yuu", "unasaka"),

    (3428370, 1, 540, "narahara_chisa", "mikoto akemi"),
    (3428370, 853, 1426, "hosaki_sumire", "unasaka"),
    (3428371, 1, 525, "futakami_youko", "mikoto akemi"),
    (3428371, 526, 1239, "yaegashi_yuuki", "unasaka"),

    (3428373, 235, 495, "kawagiri_hana", "unasaka"),
    (3428373, 853, 1107, "shinjou_yuu", "unasaka"),
    (3428373, 55, 234, "narahara_chisa", "mikoto akemi"),
    (3428373, 584, 747, "hosaki_sumire", "unasaka"),
    (3428373, 748, 852, "futakami_youko", "mikoto akemi"),
    (3428373, 1108, 1311, "yaegashi_yuuki", "unasaka"),
    # Mell Kiss
    (1201411, 4, 91, "shirashima_airi", "unasaka"),
    (1201411, 92, 209, "kagura_kaede", "unasaka"),
    (1201411, 210, 307, "akitsuki_tsuduri", "hanekoto"),
    (1201411, 308, 444, "miyamori_yuzu", "hanekoto"),
    (1201411, 445, 453, "shirashima_airi", "unasaka"),
    (1201411, 454, 467, "kagura_kaede", "unasaka"),
    (1201411, 468, 481, "akitsuki_tsuduri", "hanekoto"),
    (1201411, 482, 492, "miyamori_yuzu", "hanekoto"),

    (3428372, 1, 301, "shirashima_airi", "unasaka"),
    (3428372, 490, 912, "kagura_kaede", "unasaka"),
    (3428372, 967, 1309, "akitsuki_tsuduri", "hanekoto"),
    (3428372, 1310, 1638, "miyamori_yuzu", "hanekoto"),
    # Fuyukiss
    (1965006, 3, 96, "amari_mito", "unasaka"),
    (1965006, 97, 214, "shinomiya_touka", "unasaka"),
    (1965006, 215, 326, "mikasa_yuki", "unasaka"),
    # Love Clear
    (1354716, 12, 137, "masaki_eri", "mikoto akemi"),
    (1354716, 138, 300, "amano_himari", "mikoto akemi"),
    (1354716, 301, 435, "chihiro_kohane", "hitsuji takako"),
    (1354716, 436, 562, "misumi_ritsu", "niro"),
    (1354716, 563, 570, "masaki_eri", "mikoto akemi"),
    (1354716, 571, 575, "amano_himari", "mikoto akemi"),
    (1354716, 576, 577, "chihiro_kohane", "hitsuji takako"),
    (1354716, 578, 582, "misumi_ritsu", "niro"),

    (1365682, 1, 1792, "masaki_eri", "mikoto akemi"),
    (1365682, 1793, 2000, "amano_himari", "mikoto akemi"),
    (1366218, 1, 1276, "amano_himari", "mikoto akemi"),
    (1366218, 1709, 2000, "chihiro_kohane", "hitsuji takako"),
    (1366237, 1, 2000, "chihiro_kohane", "hitsuji takako"),
    (1366759, 1, 1100, "chihiro_kohane", "hitsuji takako"),
    (1366759, 1101, 2000, "misumi_ritsu", "niro"),
    (1366781, 1, 332, "misumi_ritsu", "niro"),
    # Kanojo to Ore to Koibito to
    (656567, 2, 228, "mihagino_ayano", "marui"),
    (656567, 229, 351, "kozomi_chikage", None),
    (656567, 352, 471, "mihagino_konoka", None),
    (656567, 472, 490, "matsugami_susuki", "marui"),
    (656567, 491, 609, "hakuto_tsukushi", "non"),
    (656567, 679, 737, "tokuyoshi_yuuko", "marui"),

    (1067655, 2, 221, "mihagino_ayano", "marui"),
    (1067655, 222, 333, "kozomi_chikage", None),
    (1067655, 334, 450, "mihagino_konoka", None),
    (1067655, 451, 534, "matsugami_susuki", "marui"),
    (1067655, 535, 666, "hakuto_tsukushi", "non"),
    (1067655, 667, 808, "tokuyoshi_yuuko", "marui"),

    (548728, 1, 55, "matsugami_susuki", "marui"),
    (3153055, 4, 171, "mihagino_ayano", "marui"),
    # Koisuru Natsu no Last Resort
    (2868164, 1, 127, "kouzaki_umi", "marui"),
    (688251, 884, 1224, "kouzaki_umi", "marui"),
    # Ecchi de Ichizu na Doinaka Nii-sama to, Koshikiyukashii Byoujaku Imouto
    (3628477, 1, 2000, "mirai_asumi", "k-ko"),
    (3433215, 1, 2000, "mirai_asumi", "k-ko"),
    (3316550, 1, 2000, "mirai_asumi", "k-ko"),
    (3316538, 1, 2000, "mirai_asumi", "k-ko"),
    (3299012, 1, 2000, "mirai_asumi", "k-ko"),
    # Gaman ga Dekinai Doutei Aniki to Sunao ni Narenai Hankou Imouto
    (2281479, 2, 951, "chie_(hankou_imouto)", "k-ko"),
    (2315112, 1, 2000, "chie_(hankou_imouto)", "k-ko"),
    (2340560, 1, 2000, "chie_(hankou_imouto)", "k-ko"),
    (2366290, 1, 2000, "chie_(hankou_imouto)", "k-ko"),
    (2392608, 1, 2000, "chie_(hankou_imouto)", "k-ko"),
    (2419989, 1, 161, "chie_(hankou_imouto)", "k-ko"),
    # Goshujin-sama, Maidfuku o Nugasanaide.
    (981306, 3, 356, "zaizen_chinatsu", "kakao"),
    (981306, 357, 706, "kurosaki_rika", "kakao"),
    (981306, 707, 1027, "naruse_koko", "kakao"),
    (981306, 1028, 1277, "ikuyama_karen", "olive"),
    # Koiyasumi 
    (1994830, 1, 371, "inaba_usaki", "mayusaki yuu"),
    (2095144, 1, 399, "inaba_usaki", "mayusaki yuu"),
    (2439947, 1, 207, "inaba_usaki", "mayusaki yuu"),
    # aotsu karin
    (1696991, 1, 2000, "uesaka_shiori", "aotsu karin"),
    (2267021, 1, 56, "mazaki_chisa", "aotsu karin"),
    (2267021, 112, 219, "mazaki_chisa", "aotsu karin"),
    # LOVE MAJYO
    (651542, 2, 356, "mikado_ichika", "kiduki erika"),
    (651542, 405, 468, "yawata_hinano", "kiduki erika"),
    (651542, 499, 585, "shinomiya_ririne", "yadapot"),
    (651542, 586, 634, "kannonzaki_nagi", "kiduki erika"),
    (651542, 635, 641, "mikado_ichika", "kiduki erika"),
    (651542, 642, 670, "kannonzaki_nagi", "kiduki erika"),
    # Yuyukana
    (3677117, 2, 82, "yuyuzuki_ako", "mitha"),
    (3677117, 83, 214, "takasaki_honoka", "mitha"),
    (3677117, 215, 304, "kusunoki_kukune", "mitha"),
    (3677117, 305, 370, "himezono_risa", "mitha"),

    (414131, 2, 80, "yuyuzuki_ako", "mitha"),
    (414131, 81, 172, "takasaki_honoka", "mitha"),
    (414131, 173, 260, "kusunoki_kukune", "mitha"),
    (414131, 261, 323, "himezono_risa", "mitha"),

    (3677354, 1, 402, "yuyuzuki_ako", "mitha"),
    (3677354, 790, 1022, "kusunoki_kukune", "mitha"),
    (3677354, 1023, 1299, "takasaki_honoka", "mitha"),
    (3677354, 1377, 1609, "kusunoki_kukune", "mitha"),
    (3677355, 33, 322, "himezono_risa", "mitha"),
    (3677354, 1, 2000, None, "mitha"),
    (3677355, 1, 2000, None, "mitha"),
    # Hanagane Kanade * Gram - Chapter:4 Ayase Kanade
    (3762625, 3, 309, "ayase_kanade", "ayuma sayu"),
    (3763168, 1, 165, "ayase_kanade", "ayuma sayu"),
    # Daunya-san to Kainushi-kun
    (3763239, 1, 454, "minamiya_ria", "unasaka"),
    (3763278, 1, 288, "minamiya_ria", "unasaka"),
    # Koakuma-chan no Yuuwaku!
    (2773875, 1, 483, "suzumori_mei", "kanekiyo miwa"),
    (3424878, 1, 288, "suzumori_mei", "kanekiyo miwa"),
    # Maid-chan wa Meido Chuu
    (3380909, 1, 193, "yuki_nana_(maid-chan_wa_meido_chuu)", "kanekiyo miwa"),
    (3393590, 1, 395, "yuki_nana_(maid-chan_wa_meido_chuu)", "kanekiyo miwa"),
    (3424877, 1, 325, "yuki_nana_(maid-chan_wa_meido_chuu)", "kanekiyo miwa"),
    # sorakoi
    (885663, 12, 160, "hikari_(sorakoi)", "miyasaka miyu"),
    (885663, 168, 174, "hikari_(sorakoi), sora_(sorakoi)", "miyasaka miyu"),
    (885663, 175, 314, "sora_(sorakoi)", "miyasaka miyu"),
    (885663, 323, 463, "airi_(sorakoi)", "miyasaka naco"),
    (885663, 464, 562, "nami_(sorakoi)", "olive"),
    # Koi ni wa Amae ga Hitsuyou Desu
    (2509733, 4, 363, "amaeda_chiwa", "go-1"),
    (2509733, 364, 626, "karakuchi_hibana", "go-1"),
    (2509733, 627, 968, "aijou_michiru", "go-1"),
    (2509733, 969, 1351, "shishikura_ouga", "go-1"),

    (2509764, 2, 247, "amaeda_chiwa", "go-1"),
    (2509764, 248, 461, "karakuchi_hibana", "go-1"),
    (2509764, 462, 654, "aijou_michiru", "go-1"),
    (2509764, 655, 885, "shishikura_ouga", "go-1"),

    (2838251, 1, 110, "shishikura_ouga", "go-1"),
    (2838251, 111, 199, "amaeda_chiwa", "go-1"),
    (2838251, 200, 302, "karakuchi_hibana", "go-1"),
    (2838251, 303, 405, "aijou_michiru", "go-1"),
    (2838251, 406, 491, "shishikura_ouga", "go-1"),
    (2838251, 492, 601, "amaeda_chiwa", "go-1"),
    (2838251, 602, 707, "karakuchi_hibana", "go-1"),
    (2838251, 708, 796, "aijou_michiru", "go-1"),

    (2888964, 1, 64, "amaeda_chiwa", "go-1"),
    (2888964, 65, 145, "karakuchi_hibana", "go-1"),
    (2888964, 146, 238, "aijou_michiru", "go-1"),
    (2888964, 239, 307, "shishikura_ouga", "go-1"),

    (3408388, 2, 606, "amaeda_chiwa", "go-1"),
    (3408388, 607, 1202, "karakuchi_hibana", "go-1"),
    (3408388, 1203, 1793, "aijou_michiru", "go-1"),
    (3408388, 1794, 2000, "shishikura_ouga", "go-1"),
    (3408406, 2, 408, "shishikura_ouga", "go-1"),

    (3425227, 2, 64, "amaeda_chiwa", "go-1"),
    (3425227, 65, 131, "karakuchi_hibana", "go-1"),
    (3425227, 132, 208, "aijou_michiru", "go-1"),
    (3425227, 209, 293, "shishikura_ouga", "go-1"),

    (3425231, 2, 75, "amaeda_chiwa", "go-1"),
    (3425231, 76, 148, "karakuchi_hibana", "go-1"),
    (3425231, 149, 233, "aijou_michiru", "go-1"),
    (3425231, 234, 291, "shishikura_ouga", "go-1"),
    # LOVEPICAL-POPPY!
    (2899735, 1, 2000, None, "hanamaru"),
    # Kohinata Yuzuki to Shoya Shitai
    (3422304, 1, 278, "kohinata_yuzuki", "go-1"),
    # Koi to H Shika Shiteinai!
    (2697069, 3, 713, "kuranosono_iwai", "go-1"),
    (2697069, 714, 1419, "kousaki_ririka", "go-1"),
    # Hoshi no Otome to Rikka no Shimai
    (2073139, 2, 73, "yamabuki_alice", "mizuki yuuma"),
    (2073139, 101, 112, "matsurika_karen", "nanotaro"),
    (2073139, 120, 174, "matsurika_karen", "nanotaro"),
    (2073139, 175, 269, "kuchinashi_nerine", "mizuki yuuma"),
    (2073139, 270, 360, "kokonoe_sumire", "mutou kurihito"),

    (2073373, 2, 73, "yamabuki_alice", "mizuki yuuma"),
    (2073373, 100, 111, "matsurika_karen", "nanotaro"),
    (2073373, 119, 173, "matsurika_karen", "nanotaro"),
    (2073373, 174, 268, "kuchinashi_nerine", "mizuki yuuma"),
    (2073373, 269, 358, "kokonoe_sumire", "mutou kurihito"),

    (2073318, 3, 95, "yamabuki_alice", "mizuki yuuma"),
    (2073318, 132, 149, "matsurika_karen", "nanotaro"),
    (2073318, 157, 223, "matsurika_karen", "nanotaro"),
    (2073318, 224, 350, "kuchinashi_nerine", "mizuki yuuma"),
    (2073318, 351, 455, "kokonoe_sumire", "mutou kurihito"),

    (3457657, 295, 672, "yamabuki_alice", "mizuki yuuma"),
    (3457657, 673, 882, "matsurika_karen", "nanotaro"),
    (3457657, 883, 1134, "kuchinashi_nerine", "mizuki yuuma"),
    (3457657, 1135, 1302, "kokonoe_sumire", "mutou kurihito"),

    (3457658, 295, 420, "yamabuki_alice", "mizuki yuuma"),
    (3457658, 421, 630, "matsurika_karen", "nanotaro"),
    (3457658, 631, 882, "kuchinashi_nerine", "mizuki yuuma"),
    (3457658, 883, 1050, "kokonoe_sumire", "mutou kurihito"),
    # Otome no Ken to Himegoto Concerto
    (2622816, 2, 218, "koshiba_anna", "mutou kurihito"),
    (2622816, 219, 247, "claire_merle", "ayase hazuki"),
    (2622816, 256, 299, "claire_merle", "ayase hazuki"),
    (2622816, 301, 410, "claire_merle", "ayase hazuki"),
    (2622816, 597, 813, "amami_iyo", "teeta.j"),
    (2622816, 814, 1064, "mukunori_riri", "eitarou"),

    (2622888, 5, 125, "koshiba_anna", "mutou kurihito"),
    (2622888, 126, 225, "claire_merle", "ayase hazuki"),
    (2622888, 327, 440, "amami_iyo", "teeta.j"),
    (2622888, 441, 578, "mukunori_riri", "eitarou"),
    (2622888, 579, 675, "koshiba_anna", "mutou kurihito"),
    (2622888, 676, 766, "claire_merle", "ayase hazuki"),
    (2622888, 852, 964, "amami_iyo", "teeta.j"),
    (2622888, 965, 1068, "mukunori_riri", "eitarou"),

    (2704363, 2, 219, "koshiba_anna", "mutou kurihito"),
    (2704363, 220, 410, "claire_merle", "ayase hazuki"),
    (2704363, 649, 837, "amami_iyo", "teeta.j"),
    (2704363, 838, 1065, "mukunori_riri", "eitarou"),

    (2628232, 4, 903, None, "mutou kurihito"),
    (2628232, 1054, 2000, "koshiba_anna", "mutou kurihito"),
    (2628229, 1, 900, "amami_iyo", "teeta.j"),
    (2628229, 901, 2000, "claire_merle", "ayase hazuki"),
    (2628215, 1, 972, "mukunori_riri", "eitarou"),
    
    (3457318, 1, 75, None, "mutou kurihito"),
    (3457318, 76, 300, "koshiba_anna", "mutou kurihito"),
    (3457318, 301, 750, "amami_iyo", "teeta.j"),
    (3457318, 751, 975, "claire_merle", "ayase hazuki"),
    (3457318, 976, 1461, "mukunori_riri", "eitarou"),
    (3457318, 1462, 2000, None, "eitarou"),

    (3457319, 1, 75, None, "mutou kurihito"),
    (3457319, 76, 300, "koshiba_anna", "mutou kurihito"),
    (3457319, 301, 750, "amami_iyo", "teeta.j"),
    (3457319, 751, 975, "claire_merle", "ayase hazuki"),
    (3457319, 976, 1461, "mukunori_riri", "eitarou"),
    (3457319, 1462, 2000, None, "eitarou"),
    # Onii-chan ni wa Zettai Ienai Taisetsu na Koto
    (557801, 1, 44, "kashiwabara_asuna", "teeta.j"),
    (557801, 53, 112, "kashiwabara_asuna", "teeta.j"),
    (557801, 139, 161, "kashiwabara_asuna", "teeta.j"),

    (1162649, 1, 350, "kashiwabara_asuna", "teeta.j"),
    # Imouto da kara Dekiru Koto, Imouto ja Nai to Dame na Koto.
    (606624, 14, 82, "takatou_iori", "teeta.j"),
    (606624, 89, 141, "takatou_iori", "teeta.j"),
    (606624, 236, 269, "takatou_iori", "teeta.j"),
    # Tenmondokei no Aria
    (887169, 3, 95, "tsukigami_aria", "izumi mahiru"),
    (887169, 100, 126, "tsukigami_aria", "izumi mahiru"),
    (887169, 135, 251, "tsukigami_aria", "izumi mahiru"),
    # Triangle Love -Apricot Fizz-
    (969657, 3, 100, "onose_ayame", "izumi mahiru"),
    (969657, 114, 196, "onose_ayame", "izumi mahiru"),
    # Koioto Se Piace
    (1407224, 1, 291, "kurumi_hana", "tonchan"),
    # Boukyaku Shitsuji to Koisuru Ojou-sama no Memoir
    (1456388, 1, 165, "chidori_hinano", "rubi-sama"),
    # Kimi to Hajimeru Dasanteki na Love Come
    (1230954, 3, 64, "sakura_nono", "sousouman"),
    (1230954, 65, 117, "teidou_shirayuki", "sousouman"),
    (1230954, 118, 177, "sakura_nono, teidou_shirayuki", "sousouman"),
    # Koiken Otome
    (549923, 49, 283, "yasukuni_akane", "tateha"),
    (549923, 301, 536, "eve_elain_austin", "tateha"),
    (549923, 537, 788, "kamishiro_touko", "tateha"),
    (549923, 789, 1022, "someya_yuzu", "tateha"),

    (699367, 2, 235, "yasukuni_akane", "tateha"),
    (699367, 253, 488, "eve_elain_austin", "tateha"),
    (699367, 489, 740, "kamishiro_touko", "tateha"),
    (699367, 741, 974, "someya_yuzu", "tateha"),

    (699458, 2, 84, "eve_elain_austin", "tateha"),
    (699458, 128, 227, "ichikura_chiharu", "tateha"),
    (699458, 228, 276, "minato_shiho", "tateha"),
    (699458, 277, 351, "someya_yuzu", "tateha"),
    (699458, 352, 417, "yasukuni_akane", "tateha"),
    (699458, 418, 499, "kamishiro_touko", "tateha"),
    (699458, 500, 540, "chiyoda_mari", "tateha"),
    (699458, 541, 551, "yasukuni_akane", "tateha"),
    (699458, 556, 565, "eve_elain_austin", "tateha"),
    (699458, 566, 574, "kamishiro_touko", "tateha"),
    (699458, 575, 583, "someya_yuzu", "tateha"),
    # Nyan Cafe Macchiato ~Neko ga Iru Cafe no Ecchi Jijou~
    (625211, 2, 139, "nekomori_mike", "yukie"),
    (625211, 140, 312, "nekoyashiki_perusha", "wori"),
    (625211, 313, 464, "nekokawa_ameri", "rubi-sama"),
    (625211, 642, 657, "nekomori_mike", "yukie"),
    (625211, 658, 694, "nekoyashiki_perusha", "wori"),
    (625211, 695, 708, "nekokawa_ameri", "rubi-sama"),

    (1680167, 3, 140, "nekomori_mike", "yukie"),
    (1680167, 141, 313, "nekoyashiki_perusha", "wori"),
    (1680167, 314, 465, "nekokawa_ameri", "rubi-sama"),
    (1680167, 626, 645, "nekomori_mike", "yukie"),
    (1680167, 646, 682, "nekoyashiki_perusha", "wori"),
    (1680167, 683, 697, "nekokawa_ameri", "rubi-sama"),
    (1680167, 1, 2000, None, " "),
    # ichizu na (shojo)
    (1832451, 1, 865, "wataribe_kyouka", "tsukimori hiro"),
    (2232786, 1, 1334, "hirohashi_runa", "tsukimori hiro"),
    # PriministAr
    (625156, 2, 130, "aki_kanoko", "matsushita makako"),
    (625156, 131, 137, "koma_koito", "motomiya mitsuki"),
    (625156, 138, 304, "tsugihana_misumi", "matsushita makako"),
    (625156, 305, 356, "tsugihana_ruruko", "motomiya mitsuki"),
    (625156, 357, 547, "enamori_senri", "motomiya mitsuki"),
    (625156, 548, 729, "kikura_shioji", "motomiya mitsuki"),
    (625156, 730, 886, "touri_tsubasa", "hatori piyoko"),

    (878230, 3, 48, "koma_kayano", "motomiya mitsuki"),
    (878230, 49, 151, "enamori_senri", "motomiya mitsuki"),

    (2566446, 2, 179, "aki_kanoko", "matsushita makako"),
    (2566446, 180, 186, "koma_koito", "motomiya mitsuki"),
    (2566446, 187, 227, "hikosaki_karen", "mitsu king"),
    (2566446, 228, 273, "koma_kayano", "motomiya mitsuki"),
    (2566446, 274, 487, "tsugihana_misumi", "matsushita makako"),
    (2566446, 571, 624, "tsugihana_ruruko", "motomiya mitsuki"),
    (2566446, 625, 959, "enamori_senri", "motomiya mitsuki"),
    (2566446, 960, 1155, "kikura_shioji", "motomiya mitsuki"),
    (2566446, 1156, 1321, "touri_tsubasa", "hatori piyoko"),
    # SuGirly Wish
    (656929, 2, 136, "kamira_akane", "sakura hanpen"),
    (656929, 137, 278, "tsukigase_anna", "rakko"),
    (656929, 279, 400, "shirosaki_hina", "sakura hanpen"),
    (656929, 401, 558, "yusa_kurumi", "rakko"),
    (656929, 559, 698, "himeyuri_megumi", "rakko"),
    # Melty Moment
    (670793, 2, 161, "ichijou_aoi", "takayaki"),
    (670793, 162, 180, None, "odawara hakone"),
    (670793, 181, 384, "fujibayashi_misao", "takayaki"),
    (670793, 385, 548, "amane_natsuki", "rakko"),
    (670793, 549, 726, "ayazaki_sumire", "odawara hakone"),
    (670793, 727, 750, "hiiragi_chiemi", "odawara hakone"),
    (670793, 751, 775, None, "takayaki"),
    (670793, 776, 968, "orie_yuuka", "rakko"),

    (734121, 1, 61, "ichijou_aoi", "takayaki"),
    (734121, 62, 87, None, "odawara hakone"),

    (743522, 2, 65, "ayazaki_sumire", "odawara hakone"),
    (743522, 66, 107, "hiiragi_chiemi", "odawara hakone"),
    # Lovely Quest
    (735369, 2, 131, "aino_thea_couvreur", "sakura hanpen"),
    (735369, 132, 252, "nishina_ayaka", "rakko"),
    (735369, 253, 373, "konose_hami", "sakura hanpen"),
    (735369, 374, 519, "yaotome_iroha", "rakko"),
    (735369, 520, 647, "sakuraba_minaho", "rakko"),
    # Strawberry Nauts 
    (899905, 60, 247, "suzunae_houmi", "motomiya mitsuki"),
    (899905, 248, 487, "yatsuka_itsuki", "hatori piyoko"),
    (899905, 488, 729, "aoto_mikamo", "motomiya mitsuki"),
    (899905, 805, 1000, "hiwa_touko", "matsushita makako"),
    (899905, 1001, 1211, "kusunoki_yao", "hatori piyoko"),

    (1321476, 59, 245, "suzunae_houmi", "motomiya mitsuki"),
    (1321476, 246, 476, "yatsuka_itsuki", "hatori piyoko"),
    (1321476, 477, 717, "aoto_mikamo", "motomiya mitsuki"),
    (1321476, 757, 949, "hiwa_touko", "matsushita makako"),
    (1321476, 950, 1159, "kusunoki_yao", "hatori piyoko"),

    (3172654, 80, 389, "suzunae_houmi", "motomiya mitsuki"),
    (3172654, 390, 717, "yatsuka_itsuki", "hatori piyoko"),
    (3172654, 718, 1076, "aoto_mikamo", "motomiya mitsuki"),
    (3172654, 1161, 1439, "hiwa_touko", "matsushita makako"),
    (3172654, 1440, 1749, "kusunoki_yao", "hatori piyoko"),
    # Kimi no Tonari de Koishiteru!
    (845981, 2, 129, "hoshino_nagisa", "motomiya mitsuki"),
    (845981, 170, 270, "chibana_ryoka", "motomiya mitsuki"),
    (845981, 271, 385, "komatsu_rina", "motomiya mitsuki"),

    (878445, 108, 372, "hoshino_nagisa", "motomiya mitsuki"),
    (878445, 373, 552, "komatsu_rina", "motomiya mitsuki"),
    (878445, 553, 852, "chibana_ryoka", "motomiya mitsuki"),

    (1432761, 2, 257, "hoshino_nagisa", "motomiya mitsuki"),
    (1432761, 337, 508, "chibana_ryoka", "motomiya mitsuki"),
    (1432761, 509, 736, "komatsu_rina", "motomiya mitsuki"),
    # Zutto Mae kara Joshi Deshita
    (1204241, 1, 166, "kazama_sena", "tsurusaki takahiro"),
    # Chiisana Kanojo no Serenade
    (640954, 2, 95, "shirasato_kaede", "asaba yuu"),
    (640954, 96, 219, "shirasato_karin", "herurun"),
    (640954, 220, 339, "motosuwa_matsuri", "tsurusaki takahiro"),
    (640954, 340, 452, "moriya_mizuka", "herurun"),
    (640954, 453, 572, "katagai_shione", "tsurusaki takahiro"),
    # shin-neko
    (1719099, 1, 230, "koharu_(shin-neko)", "niki"),
    (1876325, 1, 989, "koharu_(shin-neko)", "niki"),
    # Casablanca no Tsubomi
    (1067604, 4, 409, "shijou_kana", "niki"),
    (1067604, 410, 638, "hirayama_miku", "niki"),
    (1067604, 639, 1002, "shiina_yuri", "niki"),
    # Koi wa Mofumofu! Love Me Teddy
    (1278720, 650, 650, None, "filter_invalid"),
    (1278720, 5, 188, "tendou_arisu", "niki"),
    (1278720, 189, 287, "kinugawa_emiri", "niki"),
    (1278720, 288, 478, "kusatsu_mei", "niki"),
    (1278720, 479, 605, "ikaho_miori", "niki"),
    # Onii-chan Continue! 
    (3489547, 181, 320, "shirayuki_yuuri", "pan"),
    (3764830, 5, 81, "shirayuki_yuuri", "pan"),
    # Asuka-san wa Nabikanai
    (2409676, 1, 525, "shishidou_asuka", "sena chifuyu"),
    (3442548, 1, 111, "shishidou_asuka", "sena chifuyu"),
    # Renai Phase
    (857130, 108, 355, "izumo_kasumi_(ren'ai_phase)", "niro"),
    (857130, 356, 529, "tonami_kokoro", "usume shirou"),
    (857130, 530, 594, "amano_rei", "tomekichi"),
    (857130, 595, 770, "kagami_suzuha", "usume shirou"),
    (857130, 771, 998, "kumihama_yuki", "niro"),
    (857130, 1107, 1153, None, "filter_invalid"),
    # Hoshi no Ouji-kun
    (876432, 3, 51, "amanatsu_purin", "sakura koharu"),
    (876432, 52, 78, "aoi_ringo", "ohara tometa"),
    (876432, 79, 94, "hakase_chino", "qp:flapper"),
    (876432, 111, 135, "kamino_kokoro", "qp:flapper"),
    (876432, 136, 156, "yuri_golovnin", "ohara tometa"),
    # Tomodachi Kara Koibito e
    (2125835, 2, 215, "amakusa_hisagi", "bonnie"),
    (2125835, 216, 390, "susukawa_mizore", "bonnie"),
    (2125835, 396, 433, "amakusa_hisagi", "bonnie"),
    (2125835, 434, 475, "susukawa_mizore", "bonnie"),

    (3443190, 1, 170, "amakusa_hisagi", "bonnie"),
    (3443190, 171, 331, "susukawa_mizore", "bonnie"),
    # Berry's
    (607672, 2, 158, "morikubo_yuna", "suzuhira hiro"),
    (607672, 159, 329, "makinosawa_ena", "hashimoto takashi"),
    (607672, 330, 338, "morikubo_yuna", "suzuhira hiro"),
    (607672, 339, 368, "makinosawa_ena", "hashimoto takashi"),
    (607672, 369, 632, None, "nanao naru"),
    (607672, 633, 758, "izuno_youko", "kimizuka aoi"),
    (607672, 759, 974, "houkou_yuuka", "sakura koharu"),

    (616120, 1, 63, "morikubo_yuna", "suzuhira hiro"),
    (616120, 64, 161, "makinosawa_ena", "hashimoto takashi"),
    (616120, 162, 294, None, "nanao naru"),
    (616120, 295, 314, "izuno_youko", "kimizuka aoi"),
    (616120, 315, 424, "houkou_yuuka", "sakura koharu"),
    # Boku no Mirai wa, Koi to Kakin to
    (1369315, 5, 146, "wataya_azusa", "niro"),
    (1369315, 147, 286, "asamori_mitsuki", "nylon"),
    (1369315, 287, 429, "mahara_shiori", "kurebayashi_noe"),
    (1369315, 430, 534, "saionji_nana", "marui"),
    (1369315, 535, 556, None, "nylon"),
    (1369315, 557, 591, "kitami_rio", "niro"),

    (1376009, 1, 204, "wataya_azusa", "niro"),
    (1376009, 205, 407, "asamori_mitsuki", "nylon"),
    (1376009, 438, 638, "mahara_shiori", "kurebayashi_noe"),
    (1376009, 639, 788, "saionji_nana", "marui"),
    (1376009, 789, 813, None, "nylon"),
    (1376009, 861, 980, "kitami_rio", "niro"),
    # Innocent Girl
    (680296, 45, 129, "hinako_nanami", "nanaca mai"),
    (680296, 136, 244, "ayashiro_kagari", "nanaca mai"),
    (680296, 245, 324, "ousaka_kanae", "nanaca mai"),
    (680296, 325, 415, "midou_konoka", "nanaca mai"),

    (2905720, 45, 129, "hinako_nanami", "nanaca mai"),
    (2905720, 136, 244, "ayashiro_kagari", "nanaca mai"),
    (2905720, 245, 324, "ousaka_kanae", "nanaca mai"),
    (2905720, 325, 415, "midou_konoka", "nanaca mai"),
    # Yuki Koi Melt
    (802560, 17, 107, "reppuuji_kanon", "nanaca mai"),
    (802560, 108, 189, "unazuki_shizuri", "nanaca mai"),
    (802560, 190, 297, "himeguri_taruhi", "nanaca mai"),
    (802560, 298, 388, "tsukumo_yuki_(yuki_koi_melt)", "nanaca mai"),

    (1625298, 26, 135, "reppuuji_kanon", "nanaca mai"),
    (1625298, 136, 245, "unazuki_shizuri", "nanaca mai"),
    (1625298, 246, 391, "himeguri_taruhi", "nanaca mai"),
    (1625298, 392, 512, "tsukumo_yuki_(yuki_koi_melt)", "nanaca mai"),
    (1625298, 513, 2000, None, "filter_invalid"),
    # Pure Girl
    (999062, 41, 134, "kanadome_miyako", "nanaca mai"),
    (999062, 135, 144, "kanadome_miyako, hoshizuki_sora, mekami_suzu, kuchifusa_yogiri", "nanaca mai"),
    (999062, 145, 254, "hoshizuki_sora", "nanaca mai"),
    (999062, 255, 344, "mekami_suzu", "nanaca mai"),
    (999062, 345, 448, "kuchifusa_yogiri", "nanaca mai"),
    # Kotonoha Maichiru Natsu no Koe
    (1332042, 22, 100, "nobara_yuu", "hinata momo"),
    (1332042, 101, 176, "rinden_aoko", "hinata momo"),
    (1332042, 177, 277, "hyakka_kanade", "hinata momo"),
    (1332042, 278, 365, "manuka_kotoha", "hinata momo"),
    # Timepiece Ensemble
    (658884, 2, 54, "tsukiyono_chiara", "sesena yau"),
    (658884, 55, 82, "torigoe_sasa", "sesena yau"),
    (658884, 83, 118, "kuramae_nanami", "sesena yau"),
    (658884, 119, 162, "yanagibashi_saori", "sesena yau"),
    (658884, 163, 207, "yushima_towako", "sesena yau"),

    (1415473, 3, 100, "tsukiyono_chiara", "sesena yau"),
    (1415473, 101, 156, "torigoe_sasa", "sesena yau"),
    (1415473, 157, 225, "kuramae_nanami", "sesena yau"),
    (1415473, 226, 312, "yanagibashi_saori", "sesena yau"),
    (1415473, 313, 396, "yushima_towako", "sesena yau"),

    (1415421, 1, 1023, "tsukiyono_chiara", "sesena yau"),
    (1415421, 1024, 1975, "kuramae_nanami", "sesena yau"),
    (1415439, 120, 1367, "kuramae_nanami", "sesena yau"),
    (1415439, 1368, 2000, "yanagibashi_saori", "sesena yau"),
    (1415459, 1, 486, "yanagibashi_saori", "sesena yau"),
    (1415459, 487, 1014, "torigoe_sasa", "sesena yau"),
    (1415459, 1039, 2000, "yushima_towako", "sesena yau"),
    (1415468, 1, 486, "yushima_towako", "sesena yau"),
    # 1/2 summer
    (504259, 2, 77, "utashiro_kanami", "sesena yau"),
    (504259, 78, 148, "kuonji_sora", "sesena yau"),
    (504259, 149, 219, "kusanagi_kazuha", "sesena yau"),
    (504259, 220, 298, "kaminogi_ushio", "sesena yau"),

    (504277, 2, 77, "utashiro_kanami", "sesena yau"),
    (504277, 78, 153, "kuonji_sora", "sesena yau"),
    (504277, 154, 227, "kusanagi_kazuha", "sesena yau"),
    (504277, 228, 306, "kaminogi_ushio", "sesena yau"),
    (504277, 368, 459, "utashiro_kanami", "sesena yau"),
    (504277, 460, 558, "kuonji_sora", "sesena yau"),
    (504277, 559, 638, "kusanagi_kazuha", "sesena yau"),
    (504277, 639, 736, "kaminogi_ushio", "sesena yau"),
    # Diamic Days
    (403939, 2, 65, "hatsushiba_kiba", "sesena yau"),
    (403939, 66, 123, "himenogawa_kotora", "sesena yau"),
    (403939, 124, 185, "koboshi_renko", "sesena yau"),
    (403939, 186, 243, "shinoyama_tokiha", "sesena yau"),
    (403939, 244, 274, "himenogawa_kanaka", "sesena yau"),
    (403939, 282, 352, None, "filter_invalid"),
    (403939, 367, 2000, None, "filter_invalid"),

    (404001, 2, 65, "hatsushiba_kiba", "sesena yau"),
    (404001, 66, 123, "himenogawa_kotora", "sesena yau"),
    (404001, 124, 185, "koboshi_renko", "sesena yau"),
    (404001, 186, 243, "shinoyama_tokiha", "sesena yau"),
    (404001, 244, 274, "himenogawa_kanaka", "sesena yau"),
    (404001, 282, 299, None, "filter_invalid"),
    (404001, 314, 2000, None, "filter_invalid"),
    # Kokokara Natsu no Innocence!
    (878408, 3, 721, "hotaruzuka_arika", "sesena yau"),
    (878408, 722, 967, "shigihara_benio", "sesena yau"),
    (878408, 968, 1192, "craletta_littorio", "sesena yau"),
    (878408, 1193, 1952, "hatsuhime_iroha", "sesena yau"),
    (878408, 1953, 2000, "kumari_kotobuki", "sesena yau"),
    (878410, 1, 772, "kumari_kotobuki", "sesena yau"),
    (878410, 773, 1631, "hotaruzuka_yuno", "sesena yau"),
    # Amanosora Retrospect
    (868952, 2, 67, "oozora_himari", "hisama kumako"),
    (868952, 68, 143, "amamiya_shiron", "mitsu king"),
    (868952, 144, 208, "isora_misumi", "mitsu king"),
    (868952, 209, 287, "amahoshi", "hisama kumako"),

    (868952, 347, 486, "amahoshi", "hisama kumako"),
    (868952, 590, 787, "oozora_himari", "hisama kumako"),
    (868952, 794, 992, "isora_misumi", "mitsu king"),
    (868952, 1127, 1301, "amamiya_shiron", "mitsu king"),
    # Koitama
    (819609, 24, 63, "kagemori_kanade", "mizuki yuuma"),
    (819609, 64, 91, "hijiri_naho", "mizuki yuuma"),
    (819609, 92, 118, "serizawa_nono", "massan"),
    (819609, 119, 209, "tenjouin_aika", "massan"),
    (819609, 210, 342, "kagemori_kanade", "mizuki yuuma"),
    (819609, 343, 458, "hijiri_naho", "mizuki yuuma"),
    (819609, 459, 582, "serizawa_nono", "massan"),
    (819609, 583, 2000, None, "filter_invalid"),
    # Ryuukishi Bloody†Saga
    (1008679, 62, 62, None, "hisama kumako"),
    (1008679, 85, 176, "saria_blance", "aikawa tatsuki"),
    (1008679, 177, 290, "rize_mknest", "aikawa tatsuki"),
    (1008679, 291, 379, "mea_hartlean", "aikawa tatsuki"),
    (1008679, 380, 478, "arena_alseif", "aikawa tatsuki"),
    (1008679, 479, 527, None, "aikawa tatsuki"),
    (1008679, 1, 2000, None, "zunta"),

    (1149849, 25, 186, "saria_blance", "aikawa tatsuki"),
    (1149849, 187, 341, "rize_mknest", "aikawa tatsuki"),
    (1149849, 342, 508, "mea_hartlean", "aikawa tatsuki"),
    (1149849, 509, 676, "arena_alseif", "aikawa tatsuki"),
    (1149849, 1, 2000, None, "zunta"),

    (1418872, 1, 1400, "saria_blance", "aikawa tatsuki"),
    (1418872, 1401, 2000, "rize_mknest", "aikawa tatsuki"),
    (1418929, 1, 964, "rize_mknest", "aikawa tatsuki"),
    (1418929, 965, 2000, "mea_hartlean", "aikawa tatsuki"),
    (1419012, 1, 318, "mea_hartlean", "aikawa tatsuki"),
    (1419012, 319, 1638, "arena_alseif", "aikawa tatsuki"),
    # Nightmare×BlackCat 〜Tsuioku no Beyond〜
    (3775903, 51, 324, "mikazuki_megu", "noba"),
    (3775903, 325, 455, "amou_hibiki", "noba"),
    # Namaiki Delation
    (615331, 84, 198, "meia_krauselung_hakuhou", "syroh"),
    (615331, 199, 341, "natsushima_misaki", "syroh"),
    (615331, 342, 508, "shinkai_nagisa_(namaiki_deretion)", "syroh"),
    (615331, 509, 626, "nishimura_shiori", "syroh"),
    (615331, 627, 2000, None, "filter_invalid"),
    # Wagamama High Spec
    (931781, 2, 515, "rokuonji_kaoruko", "utsunomiya tsumire"),
    (931781, 516, 902, "sakuragi_roofolet_ashe", "utsunomiya tsumire"),
    (931781, 903, 1351, "narumi_toa", "utsunomiya tsumire"),
    (931781, 1352, 1810, "miyase_mihiro", "utsunomiya tsumire"),

    (1106715, 2, 455, "watanuki_karen", "utsunomiya tsumire"),
    (1106715, 469, 831, "takatsuka_chitose", "utsunomiya tsumire"),
    (1106715, 832, 1148, "iwakuma_yukari", "utsunomiya tsumire"),

    (1106725, 2, 362, "rokuonji_kaoruko", "utsunomiya tsumire"),
    (1106725, 363, 752, "sakuragi_roofolet_ashe", "utsunomiya tsumire"),
    (1106725, 753, 1193, "narumi_toa", "utsunomiya tsumire"),
    (1106725, 1194, 1532, "miyase_mihiro", "utsunomiya tsumire"),
    # Arui wa Koi to Iu Na no Mahou
    (1133865, 3, 157, "serea", "hasune"),
    (1133865, 158, 310, "fati", "hasune"),
    (1133865, 311, 415, "rian", "hasune"),

    (1135121, 2, 218, "serea", "hasune"),
    (1135121, 219, 478, "fati", "hasune"),
    (1135121, 479, 675, "rian", "hasune"),
    # Einstein yori Ai o Komete
    (1766047, 1, 43, "arimura_romi", "kimishima ao"),
    (1766047, 44, 97, "sakashita_iina", "kimishima ao"),
    (1766047, 98, 136, "nitta_shinobu", "kimishima ao"),
    (1766047, 137, 178, "nishino_kasumi", "kimishima ao"),

    (1766362, 33, 86, "sakashita_iina", "kimishima ao"),
    (1766362, 87, 128, "nishino_kasumi", "kimishima ao"),
    (1766362, 138, 169, "arimura_romi", "kimishima ao"),
    (1766362, 176, 214, "nitta_shinobu", "kimishima ao"),

    (2018513, 37, 45, "arimura_romi", "kimishima ao"),
    (2018513, 49, 82, "arimura_romi", "kimishima ao"),
    # Toshishita Kanojo
    (1941963, 1, 336, "ousaka_ayane", "kyou"),
    (3442586, 1, 2000, "ousaka_ayane", "kyou"),
    (3442587, 1, 2000, "ousaka_ayane", "kyou"),
    # Kimi e Okuru, Sora no Hana
    (1432735, 26, 107, "azuse_matsuri", "yukie"),
    (1432735, 108, 213, "nishizono_kanna", "yukie"),
    (1432735, 214, 298, "nasuhara_hinagiku", "yukie"),
    (1432735, 595, 607, "azuse_matsuri", "yukie"),
    (1432735, 608, 626, "nishizono_kanna", "yukie"),
    (1432735, 627, 670, "nasuhara_hinagiku", "yukie"),

    (549925, 26, 107, "azuse_matsuri", "yukie"),
    (549925, 108, 213, "nishizono_kanna", "yukie"),
    (549925, 214, 298, "nasuhara_hinagiku", "yukie"),
    (549925, 595, 607, "azuse_matsuri", "yukie"),
    (549925, 608, 626, "nishizono_kanna", "yukie"),
    (549925, 627, 670, "nasuhara_hinagiku", "yukie"),
    # midori no umi
    (844145, 2, 115, "michiru_(midori_no_umi)", "saeki hokuto"),
    (844145, 116, 218, "chisha", "yukie"),
    (844145, 219, 270, "haina", "saeki hokuto"),
    (844145, 323, 404, None, "yukie"),
    (844145, 405, 467, "sara_(midori_no_umi)", "saeki hokuto"),
    (844145, 468, 520, "tsumugi_(midori_no_umi)", "yukie"),

    (844159, 3, 194, "michiru_(midori_no_umi)", "saeki hokuto"),
    (844159, 195, 374, "chisha", "yukie"),
    (844159, 375, 482, "haina", "saeki hokuto"),
    (844159, 483, 986, None, "yukie"),
    (844159, 987, 1202, "sara_(midori_no_umi)", "saeki hokuto"),
    (844159, 1203, 1378, "tsumugi_(midori_no_umi)", "yukie"),

    (1324198, 2, 108, "michiru_(midori_no_umi)", "saeki hokuto"),
    (1324198, 109, 204, "chisha", "yukie"),
    (1324198, 205, 253, "haina", "saeki hokuto"),
    (1324198, 254, 383, None, "yukie"),
    (1324198, 384, 440, "sara_(midori_no_umi)", "saeki hokuto"),
    (1324198, 441, 489, "tsumugi_(midori_no_umi)", "yukie"),
    (1324198, 490, 2000, None, "filter_invalid"),
    # Chuunibyou na Kanojo no Love Equation
    (819681, 39, 159, "kohinata_aoi", "kaniya shiku"),
    (819681, 160, 302, "hoshino_spica_(chuunibyou_na_kanojo_no_love_equation)", "yukie"),
    (819681, 303, 428, "hanagasaki_momo", "kaniya shiku"),
    (819681, 429, 570, "kuromine_mion", "kaniya shiku"),
    (819681, 571, 669, "ayase_chisato", "yukie"),
    # Kagi o Kakushita Kago no Tori -Bird in cage hiding the key-
    (1740852, 1, 219, "kujakuseki_touko", "yukie"),
    (1740852, 220, 391, "aobazuku_mion", "yukie"),
    (1740852, 392, 569, "tsubamesawa_yoru", "yukie"),
    (1740852, 570, 743, "mizuha_iduru", "yukie"),
    (1740852, 802, 821, "kujakuseki_touko", "yukie"),
    (1740852, 822, 852, "aobazuku_mion", "yukie"),
    (1740852, 853, 875, "tsubamesawa_yoru", "yukie"),
    (1740852, 876, 897, "mizuha_iduru", "yukie"),
    # Tonari ni Kanojo no Iru Shiawase
    (1105821, 1, 106, "serizawa_chisa", "nekonyan"),
    (1188937, 1, 2000, "yukimura_shiho", "nekonyan"),
    (1403998, 1, 2000, "uryuu_koume", "nekonyan"),
    (1545838, 1, 363, "yukimura_shiho", "nekonyan"),
    (1649004, 1, 2000, "kyouno_hana", "nekonyan"),

    (1743226, 2, 151, "uryuu_koume", "nekonyan"),
    (1743226, 152, 267, "kyouno_hana", "nekonyan"),
    (1743226, 268, 413, "serizawa_chisa", "nekonyan"),
    (1743226, 414, 554, "yukimura_shiho", "nekonyan"),

    (1744527, 1, 139, "uryuu_koume", "nekonyan"),
    (1744527, 140, 255, "kyouno_hana", "nekonyan"),
    (1744527, 256, 401, "serizawa_chisa", "nekonyan"),
    (1744527, 402, 536, "yukimura_shiho", "nekonyan"),
    # Himemiya-san wa Kamaitai
    (2189428, 1, 99, "himemiya_tsubaki", "nekonyan"),
    # Kiss Kara Hajimaru Gyaru no Koi
    (2204798, 1, 2000, "hiiragi_kurumi", None),
    (2204836, 1, 2000, "hiiragi_kurumi", None),
    (2204837, 1, 2000, "hiiragi_kurumi", None),
    # Boku to Nurse no Kenshuu Nisshi
    (1204853, 1, 2000, "akagi_mio", None),
    (1389134, 1, 82, "amagi_ryou", None),
    (1389134, 83, 85, "akagi_mio", None),
    (1389134, 86, 161, "amagi_ryou", None),
    (1389134, 162, 193, "akagi_mio", None),
    # Ama Mane
    (1374097, 1, 2000, "suzumori_satsuki", "masuishi kinoto"),
    (1741737, 1, 76, "nanami_yuri", "masuishi kinoto"),
    (1741737, 77, 417, "suzumori_satsuki", "masuishi kinoto"),
    (1741737, 418, 637, "nanami_yuri", "masuishi kinoto"),
    # ChuSingura 46+1 Wacchi to Onii-chan no Love Love Nagaya Seikatsu
    (1531704, 1, 104, "yamayoshi_shinpachirou", "nui"),
    (1531704, 105, 2000, None, "filter_invalid"),
    # Keiken Zero na Classmate π
    (2047881, 1, 8, "hoshina_risa", "re"),
    (2047881, 9, 14, "miyazono_mikumo", "re"),
    (2047881, 15, 48, "hoshina_risa, miyazono_mikumo", "re"),
    (2047881, 49, 259, "hoshina_risa", "re"),
    (2047881, 260, 297, "miyazono_mikumo", "re"),
    (2047881, 620, 979, "tachibana_saki", "re"),
    # Genpei Ryouran Emaki
    (1876063, 96, 165, "kanou_roko", "hissatsukun"),
    (1876063, 168, 236, "mihishiro_shizuka", "hissatsukun"),
    (1876063, 237, 281, "kanou_roko, mihishiro_shizuka", "hissatsukun"),
    (1876063, 283, 287, "kanou_roko, mihishiro_shizuka", "hissatsukun"),
    # Sakura Mau Otome no Rondo
    (651830, 40, 168, "komine_manami", "kimishima ao"),
    (651830, 169, 332, "erihara_mitsuki", "kaniya shiku"),
    (651830, 333, 451, "amatsuka_ryouka", "nemaki"),
    (651830, 452, 635, "kadowaki_sakura", "nemaki"),
    (651830, 636, 806, "aragaki_wakana", "kaniya shiku"),

    (692291, 3, 4, "mahara_yuri", "kimishima ao"),
    (692291, 5, 28, "kadowaki_sakura", "nemaki"),
    (692291, 29, 57, "mahara_yuri", "kimishima ao"),
    (730773, 1, 82, "mahara_yuri", "kimishima ao"),

    (2277946, 1, 257, "komine_manami", "kimishima ao"),
    (2277946, 258, 449, "kadowaki_sakura", "nemaki"),
    (2277946, 450, 585, "amatsuka_ryouka", "nemaki"),
    (2277946, 586, 876, "erihara_mitsuki", "kaniya shiku"),
    (2277946, 877, 1197, "aragaki_wakana", "kaniya shiku"),
    (2277946, 1198, 1461, "mahara_yuri", "kimishima ao"),
    # Yume to Iro de Dekiteiru
    (1371164, 3, 119, "asukai_hiiro", "karory"),
    (1371164, 120, 236, "kurobane_kamome", "karory"),
    (1371164, 237, 348, "onbara_ren", "karory"),
    (1371164, 349, 451, "tokino_kumo", "karory"),
    (1371164, 452, 565, "tanabata_shiori", "karory"),
    # Mirai Radio to Jinkou Bato
    (1278685, 8, 30, "haduki_kaguya", "shimofuri"),
    (1278685, 34, 36, "haduki_kaguya", "shimofuri"),
    (1278685, 39, 418, "haduki_kaguya", "shimofuri"),
    (1278685, 419, 427, "azamino_tsubaki", "shimofuri"),
    (1278685, 430, 454, "azamino_tsubaki", "shimofuri"),
    (1278685, 457, 678, "azamino_tsubaki", "shimofuri"),
    (1278685, 679, 867, "kosumo_akina", "gyokuto_b"),
    (1278685, 868, 1059, "yamanashi_mizuki", "gyokuto_b"),

    (2470120, 6, 28, "haduki_kaguya", "shimofuri"),
    (2470120, 32, 34, "haduki_kaguya", "shimofuri"),
    (2470120, 37, 418, "haduki_kaguya", "shimofuri"),
    (2470120, 419, 427, "azamino_tsubaki", "shimofuri"),
    (2470120, 430, 454, "azamino_tsubaki", "shimofuri"),
    (2470120, 457, 678, "azamino_tsubaki", "shimofuri"),
    (2470120, 679, 865, "kosumo_akina", "gyokuto_b"),
    (2470120, 866, 1057, "yamanashi_mizuki", "gyokuto_b"),
    # World's Horniest Housewife
    (1870983, 12, 476, "kouguchi_rinko", "gyokuto_b"),
    # Gohoushi Akuma to Oshioki Tenshi
    (1838940, 5, 95, "matty_(gohoushi_akuma_to_oshioki_tenshi)", "rubi-sama"),
    (1838940, 96, 204, "luciela_valheim_melvella", "rubi-sama"),
    (1838940, 205, 295, "cial_harrell", "momoirone"),
    (1838940, 296, 381, "kohanai_touka", "maroya kayo"),
    (1838940, 382, 421, None, "maroya kayo"),
    (1838940, 422, 550, "matty_(gohoushi_akuma_to_oshioki_tenshi)", "rubi-sama"),
    (1838940, 551, 714, "luciela_valheim_melvella", "rubi-sama"),
    (1838940, 715, 906, "cial_harrell", "momoirone"),
    (1838940, 907, 1104, "kohanai_touka", "maroya kayo"),
    # Love Love Sisters ~Hanayome & Shimai-tachi to no Dokidoki Harem Seikatsu~
    (1021926, 4, 194, "nekogusa_kisara", "naenae"),
    (1021926, 195, 206, "nekogusa_kisara, sarusuberi_misatoko", "naenae"),
    (1021926, 207, 330, "sarusuberi_misatoko", "naenae"),
    (1021926, 331, 369, "nekogusa_kisara, sarusuberi_misatoko", "naenae"),
    (1021926, 370, 390, "sarusuberi_misatoko", "naenae"),
    (1021926, 391, 398, "nekogusa_kisara, sarusuberi_misatoko", "naenae"),
    (1021926, 399, 565, "pnina_maluna_ferkel", "midoriha mint"),
    (1021926, 566, 810, None, "rokudou itsuki"),
    (1021926, 811, 962, "tsurumatsu_mizuho", "ibu moina"),
    (1021926, 963, 1024, "tsukitoji_hana", "wori"),
    (1021926, 1032, 1037, None, "noda shuha"),

    (1021926, 1085, 1125, "nekogusa_kisara", "naenae"),
    (1021926, 1126, 1162, "sarusuberi_misatoko", "naenae"),
    (1021926, 1163, 1199, "pnina_maluna_ferkel", "midoriha mint"),
    (1021926, 1200, 1241, None, "rokudou itsuki"),
    (1021926, 1242, 1252, "tsurumatsu_mizuho", "ibu moina"),
    (1021926, 1253, 1253, "tsukitoji_hana", "wori"),
    # Blade x Bullet Kinrin no Soleil
    (1106239, 7, 231, None, "rubi-sama"),
    (1106239, 232, 292, None, "rokudou itsuki"),
    (1106239, 293, 364, None, "wori"),
    (1106239, 365, 419, None, "rubi-sama"),
    (1106239, 420, 534, None, "rubi-sama"),
    (1106239, 535, 594, "schwertileite_kyouka", "naenae"),
    (1106239, 595, 678, "brynhildr_arlesgrat", "naenae"),

    (1106239, 720, 752, None, "rubi-sama"),
    (1106239, 753, 778, None, "rokudou itsuki"),
    (1106239, 779, 787, None, "wori"),
    (1106239, 788, 816, None, "rubi-sama"),
    (1106239, 817, 819, "schwertileite_kyouka", "naenae"),
    (1106239, 820, 842, "brynhildr_arlesgrat", "naenae"),
    # Mousou Speaker
    (1491067, 4, 142, None, "chize"),
    (1491067, 143, 205, None, "naenae"),
    # Santaful☆Summer
    (607170, 10, 180, "kirigaya_hatsuka", "noritama"),
    (607170, 181, 349, "niieda_miyu", "usume shirou"),
    (607170, 350, 537, "nicole_(santaful_summer)", "pikazo"),
    (607170, 538, 619, "enomoto_yuri", "noritama"),
    (607170, 620, 719, "lily_abel", "pikazo"),
    (607170, 720, 849, "kleinia_burnett", " "),

    (611946, 1, 9, "kirigaya_hatsuka", "noritama"),
    (611946, 10, 35, "niieda_miyu", "usume shirou"),
    (611946, 37, 57, "nicole_(santaful_summer)", "pikazo"),

    (980642, 54, 216, "kirigaya_hatsuka", "noritama"),
    (980642, 217, 324, "kleinia_burnett", " "),
    (980642, 325, 419, "lily_abel", "pikazo"),
    (980642, 420, 579, "niieda_miyu", "usume shirou"),
    (980642, 580, 764, "nicole_(santaful_summer)", "pikazo"),
    (980642, 765, 841, "enomoto_yuri", "noritama"),
    (980642, 842, 2000, None, "filter_invalid"),

    (1114624, 1, 278, "kirigaya_hatsuka", "noritama"),
    (1114624, 279, 463, "kleinia_burnett", " "),
    (1114624, 464, 662, "lily_abel", "pikazo"),
    (1114624, 663, 798, "niieda_miyu", "usume shirou"),
    (1114624, 849, 1120, "nicole_(santaful_summer)", "pikazo"),
    (1114624, 1121, 1334, "enomoto_yuri", "noritama"),
]
# 提取目录与图片序号：.../webp/<dir>/image_<num>.webp
PATH_RE = re.compile(r"/webp/(\d+)/image_(\d+)\.webp$")
# 某些目录 id 不调整 type 字段
SKIP_TYPE_UPDATE_IDS = {793088, 1537457, 1537567, 1537491,
                        1537715, 2313627, 1805418, 2299688,
                        2299695}

# 构建优化的查找字典：目录ID -> 区间列表（保持原始顺序以维持优先级）
RANGES_DICT = {}
for d, start, end, character, artist in RANGES:
    if d not in RANGES_DICT:
        RANGES_DICT[d] = []
    RANGES_DICT[d].append((start, end, character, artist))

def lookup_targets(dir_id: int, num: int) -> Optional[Tuple[str, str]]:
    """优化后的查找函数，使用字典提高效率，保持原始列表的优先级顺序"""
    if dir_id not in RANGES_DICT:
        return None
    
    # 按RANGES原始顺序查找，前面的区间优先级更高
    for start, end, character, artist in RANGES_DICT[dir_id]:
        if end < start:
            print(f"[WARN] Invalid range for dir_id {dir_id}: start {start} > end {end}")
            continue
        if start <= num <= end:
            return character, artist
    return None


def export_ranges_to_csv(out_path: str, fallback_artists: Optional[dict] = None, fallback_counts: Optional[dict] = None) -> int:
    """将 RANGES 中的角色-画师对应关系导出为 CSV（artist 可为空），并统计次数。

    规则：
    - 角色必须存在；artist 允许为空。
    - 若角色或画师字段包含多个值（用逗号分隔）则跳过。
    - artist 为空时，若提供 fallback_artists 并能匹配到该角色的 artist，则使用该 artist。
    - 替换名称中的空格为下划线。
    - 去重并统计 count（角色、画师一致时计数累加）。
    返回写入的条目数量。
    """
    rows = {}
    for _, _, _, character, artist in RANGES:
        if not character:
            continue
        character = character.strip()
        artist = artist.strip() if artist else artist
        if not character:
            continue
        if "," in character or (artist and "," in artist):
            continue
        if (not artist) and fallback_artists:
            artist = fallback_artists.get(character)
            if artist and "," in artist:
                artist = None  # 避免多值

        character_clean = character.replace(" ", "_")
        artist_clean = (artist or "").replace(" ", "_")
        key = (character_clean, artist_clean)
        if key in rows:
            continue  # 避免重复，计数来自 fallback，不叠加

        if fallback_counts:
            cnt = fallback_counts.get(key, 1)
        else:
            cnt = 1
        rows[key] = cnt

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["character", "artist", "count"])
        writer.writerows((ch, ar, cnt) for (ch, ar), cnt in rows.items())
    return len(rows)


def export_jsonl_to_csv(input_paths: List[str], out_path: str) -> int:
    """从给定 JSONL 文件集合中提取 (character, artist) 去重导出，并统计出现次数。

    规则：
    - character 为空则跳过；artist 允许为空。
    - 若 character 或 artist 含逗号（表示多值）则跳过。
    - 名称中的空格替换为下划线。
    - 去重后写出表头 character, artist, count。
    返回写入的条目数量（行数，不是累计次数）。
    """
    counts = {}
    for path in input_paths:
        try:
            with open(path, "r", encoding="utf-8") as fin:
                for lineno, line in enumerate(fin, 1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except json.JSONDecodeError as e:
                        sys.stderr.write(f"[WARN] {path}:{lineno}: JSON decode error: {e}\\n")
                        continue

                    character = obj.get("character")
                    artist = obj.get("artist")

                    if not character:
                        continue
                    if isinstance(character, str) is False:
                        character = str(character)
                    if artist is not None and isinstance(artist, str) is False:
                        artist = str(artist)

                    character = character.strip()
                    artist = artist.strip() if artist is not None else artist

                    # 角色单值、画师多值时，将 artist 视为空再计数
                    if "," in character:
                        continue
                    if artist and "," in artist:
                        artist = ""

                    character_clean = character.replace(" ", "_")
                    artist_clean = (artist or "").replace(" ", "_")
                    key = (character_clean, artist_clean)
                    # 计数并保持插入顺序（Python3.8+ dict 有序）
                    counts[key] = counts.get(key, 0) + 1
        except FileNotFoundError:
            sys.stderr.write(f"[WARN] 输入文件 {path} 不存在，已跳过。\n")
            continue

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["character", "artist", "count"])
        writer.writerows((ch, ar, cnt) for (ch, ar), cnt in counts.items())
    return len(counts)

def process_file(in_path: str, out_path: str) -> int:
    modified = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                fout.write(line)
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                # 非法 JSON，原样写回并提示
                sys.stderr.write(f"[WARN] Line {lineno}: JSON decode error: {e}\n")
                fout.write(line)
                continue

            path = obj.get("path", " ")
            m = PATH_RE.search(path)
            if m:
                dir_id = int(m.group(1))
                num = int(m.group(2))
                targets = lookup_targets(dir_id, num)
                if targets:
                    target_character, target_artist = targets
                    # # 若设置了包含多个画师的值，标记 type 便于区分（部分目录除外）
                    # if target_artist and dir_id not in SKIP_TYPE_UPDATE_IDS:
                    #     artists_split = [a.strip() for a in target_artist.split(",") if a.strip()]
                    #     if len(artists_split) > 1:
                    #         obj["type"] = "multi_artist"
                    # 命中范围：覆盖 character 与 artist
                    if (target_character and obj.get("character") != target_character) or (target_artist and obj.get("artist") != target_artist):
                        obj["character"] = target_character if target_character else obj.get("character", " ")
                        obj["artist"] = target_artist if target_artist else obj.get("artist", " ")
                        # 若设置了包含多个画师的值，标记 type 便于区分（部分目录除外）
                        if target_artist and dir_id not in SKIP_TYPE_UPDATE_IDS:
                            artists_split = [a.strip() for a in target_artist.split(",") if a.strip()]
                            if len(artists_split) > 1:
                                obj["type"] = "multi_artist"
                            # else:
                            #      obj["type"] = "Game CG"
                        modified += 1

            # 紧凑写回，保持一行一个 JSON
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return modified

def main():
    ap = argparse.ArgumentParser(description="Set character/artist for specific image ranges in JSONL.")
    ap.add_argument("inputs", nargs="+", help="input JSONL path pattern(s), supports * wildcard; add final path as OUTPUT when not using --inplace")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input file in place")
    ap.add_argument("--export-ranges", metavar="CSV", help="export unique (character, artist) pairs from RANGES to CSV then exit")
    ap.add_argument("--ranges-fallback-csv", metavar="CSV", help="character/artist CSV used as fallback when exporting ranges and artist is empty")
    ap.add_argument("--export-jsonl", metavar="CSV", help="export unique (character, artist) pairs from JSONL inputs to CSV then exit (artist 可为空)")
    args = ap.parse_args()

    if args.export_ranges and args.export_jsonl:
        ap.error("--export-ranges 与 --export-jsonl 不能同时使用。")

    # 若仅导出范围映射或 JSONL，则不需要处理输入文件写回
    if args.export_ranges:
        fallback_artists = None
        fallback_counts = None
        if args.ranges_fallback_csv:
            fallback_artists = {}
            fallback_counts = {}
            try:
                with open(args.ranges_fallback_csv, newline="", encoding="utf-8") as fcsv:
                    reader = csv.DictReader(fcsv)
                    for row in reader:
                        ch = row.get("character")
                        ar = row.get("artist")
                        cnt = row.get("count")
                        if not ch:
                            continue
                        ch = ch.strip()
                        ar = ar.strip() if ar is not None else ""
                        if "," in ch or (ar and "," in ar):
                            continue
                        try:
                            cnt_val = int(cnt) if cnt is not None else 1
                        except ValueError:
                            cnt_val = 1
                        # 只保留第一个出现的 artist 作为回落（若存在），同时记录计数
                        if ar:
                            fallback_artists.setdefault(ch, ar)
                            ch_space = ch.replace("_", " ")
                            fallback_artists.setdefault(ch_space, ar)
                        else:
                            ch_space = ch.replace("_", " ")

                        key = (ch.replace(" ", "_"), ar.replace(" ", "_"))
                        fallback_counts.setdefault(key, cnt_val)
                        # 兼容下划线写法的查找（RANGES 中多为空格）
                        key_space = (ch_space.replace(" ", "_"), ar.replace(" ", "_"))
                        fallback_counts.setdefault(key_space, cnt_val)
            except FileNotFoundError:
                ap.error(f"回落 CSV 文件 {args.ranges_fallback_csv} 不存在。")

        written = export_ranges_to_csv(args.export_ranges, fallback_artists, fallback_counts)
        print(f"Exported {written} unique rows to {args.export_ranges}")
        return

    # 先展开输入模式（导出 JSONL 也复用）
    def expand_patterns(patterns: List[str]) -> List[str]:
        expanded: List[str] = []
        for pattern in patterns:
            if glob.has_magic(pattern):
                matches = sorted(glob.glob(pattern))
                if not matches:
                    ap.error(f"模式 {pattern} 未匹配到任何文件。")
                expanded.extend(matches)
            else:
                if not os.path.exists(pattern):
                    ap.error(f"输入文件 {pattern} 不存在。")
                expanded.append(pattern)
        return expanded

    if args.export_jsonl:
        expanded_inputs = expand_patterns(args.inputs)
        if not expanded_inputs:
            ap.error("未提供有效的输入文件。")
        written = export_jsonl_to_csv(expanded_inputs, args.export_jsonl)
        print(f"Exported {written} unique rows to {args.export_jsonl}")
        return

    patterns: List[str]
    output_path: Optional[str] = None
    if args.inplace:
        patterns = args.inputs
    else:
        if len(args.inputs) < 2:
            ap.error("非 --inplace 模式下请提供输入和输出文件，如: script in.jsonl out.jsonl")
        *patterns, output_path = args.inputs

    expanded_inputs: List[str] = expand_patterns(patterns)

    if not expanded_inputs:
        ap.error("未提供有效的输入文件。")

    if not args.inplace and len(expanded_inputs) != 1:
        ap.error("OUTPUT 仅能和一个输入文件一起使用。")

    def process_inplace(path: str) -> int:
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".jsonl_tmp_", dir=dir_, text=True)
        os.close(fd)
        try:
            changed = process_file(path, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return changed

    total_modified = 0
    if args.inplace:
        for in_path in expanded_inputs:
            changed = process_inplace(in_path)
            total_modified += changed
            print(f"{in_path}: Modified lines {changed}")
    else:
        changed = process_file(expanded_inputs[0], output_path)
        total_modified = changed
        print(f"{expanded_inputs[0]} -> {output_path}: Modified lines {changed}")

    if len(expanded_inputs) > 1:
        print(f"Total modified lines: {total_modified}")

if __name__ == "__main__":
    main()
