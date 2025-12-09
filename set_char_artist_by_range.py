#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys, tempfile, os, glob
from typing import List, Tuple, Optional

# 需要修改的区间（目录ID, 起始序号, 结束序号，角色，画师，均闭区间）
RANGES: List[Tuple[int, int, int, str, str]] = [
    (3328374, 3, 2000, None, "utsunomiya tsumire"),
    (2204943, 1, 2000, None, "taniyama-san"),
    (3417336, 1, 2000, None, "suzumori"),
    (3417337, 1, 2000, None, "suzumori"),
    (3417347, 1, 2000, None, "suzumori"),
    (3417364, 1, 2000, None, "suzumori"),
    (3417365, 1, 2000, None, "suzumori"),
    (2893243, 1, 2000, "otome_kokoro", "yuuki rika"),
    (2891511, 1, 2000, "umino_miyako", "yuuki rika"),
    (3000207, 1, 2000, None, "yuzuna hiyo"),
    (3000208, 1, 2000, None, "yuzuna hiyo"),
    (3000209, 1, 2000, None, "yuzuna hiyo"),
    (3000440, 1, 2000, None, "yuzuna hiyo"),
    (3000441, 1, 2000, None, "yuzuna hiyo"),
    (2893246, 1, 2000, "harukaze_meguri", "yuuki rika"),
    (2891509, 1, 2000, "toiro_kirame", "fuyuichi monme"),
    (727768, 345, 461, "rindo tsubame", "chikotam"),
    (727768, 630, 647, "rindo tsubame", "chikotam"),
    (727768, 3, 25, "takakura anzu", "primil"),
    (727768, 31, 106, "takakura anzu", "primil"),
    (727768, 107, 220, "takakura anri", "primil"),
    (688336, 625, 741, "rindo tsubame", "chikotam"),
    (688336, 34, 147, "takakura anri", "primil"),
    (688336, 173, 195, "takakura anzu", "primil"),
    (688336, 201, 276, "takakura anzu", "primil"),
    # Ouchi ni Kaeru made ga Marshmallow desu
    (1093883, 11, 538, "kasukabe kanon", "sasorigatame"),
    (1093883, 539, 1129, "raiha raikkonen", "ashisyun"),
    (1093883, 1130, 1829, "misuzu sasa", "chikotam"),
    (1093883, 1867, 2000, "asaka ushio", "sasorigatame"),
    (1093884, 2, 434, "asaka ushio", "sasorigatame"),
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
    (1146404, 407, 582, "kujou_shizuru", "shiramori yuse"),
    (1146404, 891, 1017, None, "shiramori yuse"),
    (1146404, 583, 773, None, "nanaroba hana"),
    (1146404, 774, 890, "ennis yutoria", "nanaroba hana"),
    (1146404, 14, 196, "konata konatsu", "kimishima ao"),
    (1146404, 197, 406, None, "kimishima ao"),
    (1409248, 1, 2000, "konata konatsu", None),
    (3000014, 35, 137, None, "kimishima ao"),
    (3000014, 220, 324, None, "shiratama"),
    (1340509, 1, 66, "konata konatsu", "kimishima ao"),
    (1340509, 67, 174, None, "kimishima ao"),
    (1340509, 175, 274, "kujou_shizuru", "shiramori yuse"),
    (1340509, 275, 338, None, "nanaroba hana"),
    (1340509, 339, 386, "ennis yutoria", "nanaroba hana"),
    (1340509, 387, 434, None, "shiramori yuse"),
    # Amairo Chocolata
    (2746430, 13, 20, "misono ichika", "shiratama"),
    (2746430, 21, 28, None, "korie riko"),
    (2746430, 29, 36, "momose kaguya", "shiratama"),
    (2746430, 37, 74, "momose mitsuki", "shiratama"),
    (2746430, 219, 259, None, "shiratama"),
    (2746430, 278, 295, "momose kaguya", "shiratama"),
    (2746430, 296, 340, "momose mitsuki", "shiratama"),
    (2746430, 341, 384, "kohana (amairo chocolata)", "korie riko"),
    (1920176, 2, 9, None, "shiratama"),
    (1920176, 10, 15, None, "korie riko"),
    (1920176, 16, 17, None, "shiratama"),
    (1920176, 18, 66, "misono ichika", "shiratama"),
    (1920176, 116, 155, "momose kaguya", "shiratama"),
    (1920176, 156, 163, "momose mitsuki, momose kaguya", "shiratama"),
    (1920176, 164, 212, "misono ichika", "shiratama"),
    (1920176, 259, 298, "momose kaguya", "shiratama"),
    (1920176, 299, 300, "amamiya mikuri, yukimura chieri", "shiratama, korie riko"),
    (1920176, 301, 320, "momose mitsuki, maiba nana", "shiratama, korie riko"),
    (1920176, 321, 380, "misono ichika", "shiratama"),
    (1920176, 429, 474, "momose kaguya", "shiratama"),
    (1562101, 50, 89, "yukimura chieri", "shiratama"),
    (1562101, 155, 196, "yukimura chieri", "shiratama"),
    (868607, 39, 141, "himeno sena", "kimishima ao"),
    (868607, 224, 328, "komari yui", "shiratama"),
    (1245707, 13, 23, "amanogawa saya", "motoi ayumu"),
    (1245707, 48, 72, "amanogawa saya", "motoi ayumu"),
    (1245707, 107, 130, "amanogawa saya", "motoi ayumu"),
    (1245707, 305, 318, "amanogawa saya", "motoi ayumu"),
    (943537, 618, 837, "amanogawa saya", "motoi ayumu"),
    (900491, 1283, 1540, "amanogawa saya", "motoi ayumu"),
    (634594, 11, 13, "futaba hisui", "nanase meruchi"),
    (634594, 17, 19, "futaba hisui", "nanase meruchi"),
    (634594, 221, 292, "futaba hisui", "nanase meruchi"),
    (522375, 46, 191, "hondou ayano", "primil"),
    (522375, 298, 440, "amamoto louis", "primil"),
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

    (633524, 2, 141, "luce yami asutarite", "yamakaze ran"),
    (633524, 142, 277, "julia lin road", "sakurazaka tsuchiyu"),
    (633524, 278, 405, "mitsu no tama yori hime", "yamakaze ran"),
    (633524, 406, 553, "amagi karin", "yamakaze ran"),
    (633524, 672, 815, "shirahase yuuna", "yamakaze ran"),

    (1217027, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (1217027, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (1217027, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (1217027, 407, 554, "amagi karin", "yamakaze ran"),
    (1217027, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (634833, 3, 142, "luce yami asutarite", "yamakaze ran"),
    (634833, 143, 278, "julia lin road", "sakurazaka tsuchiyu"),
    (634833, 279, 406, "mitsu no tama yori hime", "yamakaze ran"),
    (634833, 407, 554, "amagi karin", "yamakaze ran"),
    (634833, 673, 816, "shirahase yuuna", "yamakaze ran"),

    (1805418, 519, 793, "mihama_inori", "yuzuna hiyo"),
    (1805418, 1292, 1609, "mihama_inori", "yuzuna hiyo"),
    (1805418, 1139, 1290, "ikegai_mayu", "konomi"),
    (1805418, 1942, 2000, "ikegai_mayu", "konomi"),
    (1805418, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (1805420, 107, 388, "ikegai_mayu", "konomi"),
    # Himawari!! -Anata Dake wo Mitsumeteru-
    (735531, 274, 380, "mikazuki tenma (himawari)", None),
    # Docchi no i ga Suki Desu ka?
    (1525889, 605, 1190, "tanemura koyuzu", "netarou"),
    # Kujiragami no Tearstilla
    (885411, 6, 98, "tenkawa_mitsuki", None),
    (3450027, 1, 552, "tenkawa_mitsuki", "mikagami mamizu"),
    # Floral Flowlove
    (960177, 3, 207, "adelheid_von_bergstrasse", "matsumiya kiseri"),
    (960177, 208, 408, "mihato kano", "hontani kanae"),
    (960177, 409, 647, "tsubaki kohane", "arisue tsukasa"),
    (960177, 648, 890, "tokisaka nanao", "toranosuke"),
    # Kiniro Loveriche -Golden Time-
    (1953903, 2, 323, "sylvia_le_cruzcrown_sortilege_sisua", "hontani kanae"),
    (1953903, 324, 543, "kisaki_reina", "toranosuke"),
    (1936004, 771, 1089, "sylvia_le_cruzcrown_sortilege_sisua", "hontani kanae"),
    (1936004, 391, 605, "kisaki_reina", "toranosuke"),
    (1369330, 37, 249, "jougasaki_ayaka", "arisue tsukasa"),
    (2274092, 34, 246, "jougasaki_ayaka", "arisue tsukasa"),
    # Hanayome to Maou
    (746122, 9, 67, "celica tepes lunatica", None),
    # Anata ni Koisuru Renai Recette
    (1243251, 1, 1280, "tachibana nonoka", "komeshiro kasu"),
    (1243251, 1281, 2000, "oozono yuzuki", "fummy"),
    (1243267, 1, 560, "oozono yuzuki", "fummy"),
    (1243267, 561, 1840, "kagiyoshi fuuka", "komeshiro kasu"),
    (1243267, 1841, 2000, "shirosaki mieru", "pero"),
    (1243602, 1, 1120, "shirosaki mieru", "pero"),
    (1067390, 23, 182, "tachibana nonoka", "komeshiro kasu"),
    (1067390, 183, 338, "oozono yuzuki", "fummy"),
    (1067390, 339, 498, "kagiyoshi fuuka", "komeshiro kasu"),
    (1067390, 499, 706, "shirosaki mieru", "pero"),
    # Tsumi no Hikari Rendezvous Goukaban
    (913381, 1, 248, "tsubaki fuuka", "satasama"),
    (913368, 773, 1169, "tsubaki fuuka", "satasama"),
    (913368, 1844, 2000, "tsubaki fuuka", "satasama"),
    (2657576, 774, 1170, "tsubaki fuuka", "satasama"),
    (2657600, 409, 813, "tsubaki fuuka", "satasama"),
    (2657600, 1, 408, "masumi_ai", "yuzuna hiyo"),
    (2657600, 814, 1142, "misono_tsubura", "mizuno sao"),
    # Amatsutsumi
    (2285016, 334, 390, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (2285016, 972, 1130, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (1121856, 565, 1574, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (959791, 726, 940, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
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
    (970538, 1, 103, "tsukimiya asuka", None),
    (970538, 125, 128, "miake hiyoko", None),
    (970538, 132, 145, "tsukimiya asuka", None),
    (970538, 146, 216, "miake hiyoko", None),
    # Pure Song Garden!
    (1081513, 2, 236, "shimokuni asuka", "bekotarou"),
    (1081513, 237, 578, "hoshino iroha", "motoi ayumu"),
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
    (2560932, 73, 139, "arise_kaguya", "hontani kanae"),
    (2595303, 160, 215, "ishikawa_yae", "toranosuke"),
    # Kakenuke Seishun Sparking!
    (1719088, 6, 198, "kohinata_hibiki", "hontani kanae"),
    (1719088, 340, 531, "kaidou_nagiko", "arisue tsukasa"),
    (1719088, 198, 343, "kashima_riri", "toranosuke"),

    (3097843, 5, 197, "kohinata_hibiki", "hontani kanae"),
    (3097843, 508, 699, "kaidou_nagiko", "arisue tsukasa"),
    (3097843, 646, 791, "kashima_riri", "toranosuke"),
    # Primal x Hearts 2
    (1939336, 1, 2000, "alicetia wallenberg kezouji", "sasorigatame"),
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
    (1183093, 39, 442, "tenjindaira haruhi", "sasorigatame"),
    (1183093, 443, 858, "kanna kana", "ashisyun"),
    (1183093, 874, 1416, "kuragano sara", "sasorigatame"),
    (1183093, 1417, 1795, "komagata yuzuki", "ashisyun"),
    (868964, 612, 2000, "kuragano sara", "sasorigatame"),
    # Zettai Saikyou ☆ Oppai Sensou!!
    (536888, 348, 448, "kotone (zettai saikyou)", "any, annie"),
    (1481653, 352, 452, "kotone (zettai saikyou)", "any, annie"),
    # Amatarasu Riddle Star -
    (1033787, 2, 234, "ai_(amatarasu_riddle_star)", "syroh"),
    (1033787, 541, 731, "hatta_madori", "syroh"),
    (1033787, 776, 1118, "yukishiro miu", "any, annie"),
    (1033787, 1119, 1202, "arisu_rina", "annie"),
    (1033787, 1203, 1422, "arisu_yua", "2-g"),

    (1499212, 3, 235, "ai_(amatarasu_riddle_star)", "syroh"),
    (1499212, 542, 732, "hatta_madori", "syroh"),
    (1499212, 777, 1119, "yukishiro miu", "any, annie"),
    (1499212, 1120, 1202, "arisu_rina", "annie"),
    (1499212, 1203, 1421, "arisu_yua", "2-g"),

    (1115664, 1327, 1527, "ai_(amatarasu_riddle_star)", "syroh"),
    (1115664, 1528, 1666, "hatta_madori", "syroh"),
    (1115664, 805, 1326, "yukishiro miu", "any, annie"),
    (1115664, 1744, 1775, "arisu_rina", "annie"),
    (1115664, 1, 543, "arisu_yua", "2-g"),
    # Zettai Junshu New Kozukuri World
    (1008830, 361, 440, "tadokoro_minami", "2-g"),
    (1008830, 441, 530, "kasugai_noa", "2-g"),
    # Zettai Seifuku 
    (799866, 24, 149, "urushino_himeko", "sukoyaka gyuunyuu"),
    (799866, 758, 933, "uwaba_shiera", "araiguma"),
    # Yuuwaku Scramble
    (970288, 303, 593, "hoshimi yuki", "hinata nao"),
    # Imouto Paradise!
    (600053, 2, 200, "nanase momoka", "itou life"),
    (600053, 201, 360, "nanase ririna", "itou life"),
    (600053, 491, 671, "nanase chiharu", "itou life"),
    (600053, 672, 880, "nanase shizuku", "itou life"),
    (600053, 1118, 1150, "nanase momoka", "itou life"),
    (600053, 1151, 1183, "nanase ririna", "itou life"),
    (600053, 1229, 1257, "nanase chiharu", "itou life"),
    (600053, 1258, 1306, "nanase shizuku", "itou life"),
    (1158457, 2, 259, "nanase momoka", "itou life"),
    (1158457, 260, 465, "nanase ririna", "itou life"),
    (1158457, 626, 878, "nanase chiharu", "itou life"),
    (1158457, 879, 1159, "nanase shizuku", "itou life"),
    (1977150, 13, 1452, "nanase momoka", "itou life"),
    (1977150, 1453, 2000, "nanase ririna", "itou life"),
    (1977160, 1, 892, "nanase ririna", "itou life"),
    (1977178, 1317, 2000, "nanase chiharu", "itou life"),
    (1977201, 1, 600, "nanase shizuku", "itou life"),
    # Amanatsu Adolescence
    (1043759, 2, 22, "hyuuga_natsu", "hisama kumako"),
    (1043759, 23, 32, "sasha_mayakovskaya", "hitsuji takako"),
    (1043759, 88, 161, "hyuuga_natsu", "hisama kumako"),
    (1043759, 162, 230, "sasha_mayakovskaya", "hitsuji takako"),
    # Shougun-sama wa Otoshigoro
    (1175803, 10, 121, "tokugawa_muneharu", "shona mitsuishi"),
    (1175803, 329, 478, "tokuda_yoshimune", "shona mitsuishi"),
    # sousaku kanojo no renai koushiki
    (3425319, 1, 432, "ayase_aisa", None),
    (3425319, 433, 1197, "tsukimizaka_kiriha", None),
    (2351500, 2, 10, "ayase_aisa", None),
    (2351500, 42, 81, "ayase_aisa", None),
    (2351500, 11, 20, "tsukimizaka_kiriha", None),
    (2351500, 82, 107, "tsukimizaka_kiriha", None),
    (2070784, 19, 36, "ayase_aisa", None),
    (2070784, 51, 53, "ayase_aisa", None),
    (2070784, 185, 210, "ayase_aisa", None),
    (2070784, 762, 800, "ayase_aisa", None),
    (2070784, 815, 956, "ayase_aisa", None),
    (2070784, 54, 102, "tsukimizaka_kiriha", None),
    (2070784, 429, 603, "tsukimizaka_kiriha", None),
    # Shukufuku no Kanenone wa, Sakurairo no Kaze Totomoni
    (1321028, 2, 112, "ootori_maria", "anapom"),
    (1321028, 230, 362, "kitazono_saya", "anapom"),
    # Kokoro ga Tsunagu Koi Shirube
    (1322592, 5, 349, "kujou_himeno", None),
    # Koi wa Yumemiru Mouretsu Girl!
    (1009125, 7, 384, "mioka_aoi", "naenae"),
    (1009125, 385, 669, "mioka_aoi", "naenae"),
    (1009125, 673, 1060, "yuunagi_shizuku", "naenae"),
    (1009125, 1305, 1368, "chie", "niki"),
    (1257428, 30, 163, "mioka_aoi", "naenae"),
    (1257428, 164, 256, "mioka_aoi", "naenae"),
    (1257428, 257, 353, "yuunagi_shizuku", "naenae"),
    # Garudoma
    (2653109, 3, 115, "fuyusaki_aiko", None),
    (2653109, 873, 987, "fuyusaki_aiko", None),
    (2653109, 1310, 1378, "fuyusaki_aiko", None),
    (2836574, 3, 236, "fuyusaki_aiko", None),
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
    # Ama Koi Syrups
    (790794, 2, 181, "watanuki_tsuyuri", "pan"),
    (790794, 526, 699, "kusaka_hozumi", "pan"),
    (1166516, 2, 181, "watanuki_tsuyuri", "pan"),
    (1166516, 527, 700, "kusaka_hozumi", "pan"),
    (1166649, 2, 153, "watanuki_tsuyuri", "pan"),
    (1166649, 447, 598, "kusaka_hozumi", "pan"),
    # Tenshi☆Souzou RE-BOOT!
    (2536708, 3, 466, "shirayuki_noa", "kobuichi"),
    (2536708, 467, 708, "ozato_fumika", "hadumi rio"),
    (2537215, 410, 736, "kohibari_kurumi", "muririn"),
    (2537215, 737, 1214, "hoshikawa_kaguya", "kobuichi"),
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
    # Limelight Lemonade Jam
    (3556090, 1, 2000, "shimakoshi_tsukimi", None),
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
    (3556156, 1, 2000, "saen_nayuka", "hadumi rio"),
    (3556157, 1, 2000, "saen_nayuka", "hadumi rio"),
    (3556094, 1, 2000, "futamihara_ririko", "muririn"),
    (3556095, 1, 2000, "futamihara_ririko", "muririn"),
    (3556096, 1, 2000, "futamihara_ririko", "muririn"),
    (3556097, 1, 2000, "futamihara_ririko", "muririn"),
    (3553799, 22, 62, "harumi_ena", "muririn"),
    (3553799, 88, 129, "shimakoshi_tsukimi", None),
    (3553799, 130, 174, "futamihara_ririko", "muririn"),
    (3553799, 175, 189, "koishikawa_miku", "kobuichi"),
    (3553799, 190, 208, "saen_nayuka", "hadumi rio"),

    (3638988, 3, 1008, "harumi_ena", "muririn"),
    (3590156, 2, 604, "shimakoshi_tsukimi", None),
    (3590156, 605, 1321, "futamihara_ririko", "muririn"),
    (3590156, 1322, 1581, "koishikawa_miku", "kobuichi"),
    (3638988, 1467, 1890, "saen_nayuka", "hadumi rio"),

    # cafe stella to shinigami no chou
    (1538399, 1, 2000, "akizuki_kanna", "kobuichi"),
    (1538355, 1, 2000, "shiki_natsume", "muririn"),
    (1538430, 1, 2000, "sumizome_nozomi", "muririn"),
    (1538498, 1, 1044, "shioyama_suzune", "muririn"),

    (1522825, 3, 145, "akizuki_kanna", "kobuichi"),
    (1522825, 146, 223, "shiki_natsume", "muririn"),
    (1522825, 224, 265, "sumizome_nozomi", "muririn"),
    (1522825, 313, 382, "shioyama_suzune", "muririn"),

    (1536430, 12, 512, "akizuki_kanna", "kobuichi"),
    (1536430, 513, 1134, "shiki_natsume", "muririn"),
    (1536430, 1135, 1778, "sumizome_nozomi", "muririn"),
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
    (1189877, 179, 394, "kosimizu_kasumi", None),
    (1263294, 3, 166, "yuuki_ayaka", None),
    (1990347, 417, 568, "yuuki_ayaka", None),
    (1990347, 200, 415, "kosimizu_kasumi", None),
    (2566350, 7, 159, "yuuki_ayaka", None),
    (2566350, 177, 392, "kosimizu_kasumi", None),
    # Houkago Cinderella
    # FLIP＊FLOP
    (2362035, 1, 2000, "io_(flip_flop)", None),
    # Pure x Connect
    (820343, 185, 342, "shinozaki_ayumi_(pure_x_connect)", None),
    (820343, 548, 726, "makihara_shiho_(pure_x_connect)", None),
    # DRACU-RIOT!
    (875699, 1, 736, "yarai_miu", "muririn"),
    (875699, 1747, 2000, "inamura_rio", "kobuichi"),
    (875672, 1, 295, "inamura_rio", "kobuichi"),
    (875672, 296, 1031, "elena_olegovna_owen", "kobuichi"),
    (875672, 1032, 1335, "nicola_cepheus", "muririn"),
    # Senren*banka
    (3442432, 1, 2000, "tomotake_yoshino", "kobuichi"),
    (1890822, 1, 994, "tomotake_yoshino", "kobuichi"),
    (960624, 3, 646, "tomotake_yoshino", "kobuichi"),
    # Sanoba Witch
    (3424478, 1, 2000, "ayachi_nene", "muririn"),
    (3424479, 1, 2000, "ayachi_nene", "muririn"),
    (3424480, 1, 2000, "ayachi_nene", "muririn"),
    (3424414, 1, 2000, "togakushi_touko", "kobuichi"),
    (3424415, 1, 2000, "togakushi_touko", "kobuichi"),
    (798685, 408, 958, "togakushi_touko", "kobuichi"),
    (798679, 3, 934, "ayachi_nene", "muririn"),
    (798679, 935, 1580, "inaba_meguru", "muririn"),
    # RIDDLE JOKER
    (1541162, 1, 2000, "mitsukasa_ayase", "muririn"),
    (1543784, 1, 2000, "arihara_nanami", "kobuichi"),
    (1543991, 1, 2000, "shikibe_mayu", "muririn"),
    (1544108, 1, 2000, "nijouin_hazuki", "kobuichi"),
    (1468670, 1, 972, "mitsukasa_ayase", "muririn"),
    (1468670, 973, 1815, "arihara_nanami", "kobuichi"),
    (1468670, 1816, 2000, "shikibe_mayu", "muririn"),
    (1468698, 1, 474, "shikibe_mayu", "muririn"),
    (1468698, 475, 976, "nijouin_hazuki", "kobuichi"),
    # Amairo IsleNauts
    (607261, 2, 31, "shirley_warwick", "kobuichi"),
    (607261, 32, 63, "amagiri_yune", "muririn"),
    (607261, 64, 99, "shiraga_airi", "kobuichi"),
    (607261, 100, 125, "masaki_gaillard", "muririn"),
    (607261, 156, 187, "tia_hohenwerfen", "muririn"),

    (614344, 2, 233, "shirley_warwick", "kobuichi"),
    (614344, 234, 505, "amagiri_yune", "muririn"),
    (614344, 506, 697, "shiraga_airi", "kobuichi"),
    (614344, 698, 840, "masaki_gaillard", "muririn"),
    (614344, 841, 957, "hinomiya_konoka", "kobuichi"),
    (614344, 958, 1111, "tia_hohenwerfen", "muririn"),
    # Southern Cross Love Song / Minamijuujisei Renka
    (743876, 4, 295, "fujina_kanori", None),
    (743876, 515, 719, "naraoka_mitsuki", None),
    # Sorceress*Alive!
    (1354083, 3, 14, "akina_randal", "shona mitsuishi"),
    (1354083, 44, 51, "azuria_newfield", "shona mitsuishi"),
    (1354083, 15, 32, "yuzuriha_serval", "hayakawa halui"),
    (1354083, 200, 282, "akina_randal", "shona mitsuishi"),
    (1354083, 283, 361, "azuria_newfield", "shona mitsuishi"),
    (1354083, 420, 479, "yuzuriha_serval", "hayakawa halui"),
    # Ren'ai, Hajimemashite
    (3255903, 1, 177, "tenshi-chan_(ren'ai_hajimemashite) ", "fuyuichi monme"),
    (3255903, 180, 306, "aizawa_yukari", "unasaka"),
    (3255903, 567, 604, None, "unasaka"),
    (3554542, 2, 63, "tenshi-chan_(ren'ai_hajimemashite) ", "fuyuichi monme"),
    (3554542, 64, 110, "aizawa_yukari", "unasaka"),
    (3554542, 258, 262, "tenshi-chan_(ren'ai_hajimemashite) ", "fuyuichi monme"),
    (3554542, 263, 268, "aizawa_yukari", "unasaka"),
    # Koibana Ren'ai
    (2872360, 3, 40, "otome_kokoro", "yuuki rika"),
    (2872360, 81, 108, "harukaze_meguri", "yuuki rika"),
    (2692612, 2, 186, "otome_kokoro", "yuuki rika"),
    (2692612, 333, 459, "harukaze_meguri", "yuuki rika"),
    # Futamata Ren'ai
    (3457068, 2, 149, "nobuta_yua", "fuyuichi monme"),
    (2891508, 1, 2000, "nobuta_yua", "fuyuichi monme"),
    (2412643, 4, 57, "nobuta_yua", "fuyuichi monme"),
    (2412643, 58, 128, "toiro_kirame", "fuyuichi monme"),
    (2205648, 2, 159, "nobuta_yua", "fuyuichi monme"),
    (2205648, 450, 605, "umino_miyako", "yuuki rika"),
    (2311617, 79, 120, "umino_miyako", "yuuki rika"),
    # Renai, Karichaimashita
    (1453395, 4, 189, "segawa_emi", "fuyuichi monme"),
    (1453395, 192, 344, "tenma_hasumi", "fuyuichi monme"),
    (2043589, 3, 188, "segawa_emi", "fuyuichi monme"),
    (2043589, 191, 343, "tenma_hasumi", "fuyuichi monme"),
    (3536064, 1, 626, "segawa_emi", "fuyuichi monme"),
    (3536064, 627, 2000, "tenma_hasumi", "fuyuichi monme"),
    (1531243, 1, 115, "tenma_hasumi", "fuyuichi monme"),
    (1562392, 4, 49, "segawa_emi", "fuyuichi monme"),
    (1562392, 50, 89, "tenma_hasumi", "fuyuichi monme"),
    (1562392, 90, 118, "segawa_emi, tenma_hasumi", "fuyuichi monme"),
    (1562392, 120, 120, "segawa_emi, tenma_hasumi", "fuyuichi monme"),
    # Renai x Royale
    (1786483, 401, 545, "amagamine_renna", "yuuki rika"),
    (1786483, 573, 590, "iyori ao", "yuuki rika"),
    (1786483, 608, 671, "kagaya_yuna", "yuuki rika"),
    (1813075, 373, 1470, "amagamine_renna", "yuuki rika"),
    (1813075, 1471, 1766, "kagaya_yuna", "yuuki rika"),
    (1813075, 1767, 2000, "iyori ao", "yuuki rika"),
    (1813082, 1, 11, "iyori ao", "yuuki rika"),
    (1877188, 43, 105, "amagamine_renna", "yuuki rika"),
    (1877188, 106, 139, "kagaya_yuna", "yuuki rika"),
    (1921945, 93, 130, "iyori ao", "yuuki rika"),
    (1921951, 99, 136, "iyori ao", "yuuki rika"),
    # Sorairo Innocent
    (882267, 3, 90, "tsukigase_mahiru", "unasaka"),
    (882267, 91, 157, "tsubaki_ami", "unasaka"),
    (1420499, 2, 631, "tsukigase_mahiru", "unasaka"),
    (1420499, 632, 1035, "tsubaki_ami", "unasaka"),
    # Kanojo to Ore no Lovely Day
    (1134208, 1, 192, "mashiro yuka", "chikotam"),
    (1134208, 193, 384, "kongou alice", "chikotam"),
    (1023186, 3, 186, "mashiro yuka", "chikotam"),
    (1023186, 725, 768, "mashiro yuka", "chikotam"),
    (1023186, 187, 359, "kongou alice", "chikotam"),
    (1023186, 769, 806, "kongou alice", "chikotam"),
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
    # Sakura Iro, Mau Koro ni
    (1389160, 7, 118, None, "lucie"),
    (1389160, 119, 296, None, "yuzuka"),
    (1389160, 297, 434, "tomari_mariko", "komeshiro kasu"),
    (1389160, 435, 586, "mizushiro_mina", "yuzuka"),
    (1389160, 591, 744, None, "anapom"),
    (1445230, 1, 400, "mizushiro_mina", "yuzuka"),
    (1445230, 401, 720, None, "yuzuka"),
    (1445230, 721, 1264, "tomari_mariko", "komeshiro kasu"),
    (1445230, 1265, 1500, None, "anapom"),
    (1445272, 1, 864, None, "lucie"),
    (1445272, 865, 1080, "hino_yuki", "yuzuka"),
    # Goshujin-sama, Seira ni Yume Mitai na Icha Love Gohoushi Sasete Itadakemasu ka
    (3034052, 1, 2000, "seira_(rubi-sama)", "rubi-sama"),
    (2272848, 1, 2000, "seira_(rubi-sama)", "rubi-sama"),
    # Wan Nyan ☆ A La Mode!
    (887743, 3, 60, "nekohana_korone", "naenae"),
    (887743, 246, 300, "inuta_hana", "naenae"),
    (1131217, 3, 81, "nekohana_korone", "naenae"),
    (1131217, 467, 555, "inuta_hana", "naenae"),
    (1886653, 2, 89, "nekohana_korone", "naenae"),
    (1886653, 407, 525, "inuta_hana", "naenae"),
    (1735897, 29, 248, "nekohana_korone", "naenae"),
    (1735897, 865, 1044, "inuta_hana", "naenae"),
    # Love Love ♥ Princess
    (839209, 3, 213, "marigold_bruette_erland", "rubi-sama"),
    (839209, 214, 233, "marigold_bruette_erland,  anastasia_imperator_erland", "wori, rubi-sama"),
    (839209, 234, 432, "anastasia_imperator_erland", "wori"),
    (839209, 433, 592, "tsukimori_mio_erland", "wori"),

    (839731, 3, 407, "marigold_bruette_erland", "rubi-sama"),
    (839731, 408, 719, "anastasia_imperator_erland", "wori"),
    (839731, 720, 989, "tsukimori_mio_erland", "wori"),
    # Love Love Life
    (688579, 2, 124, "akemiya_sakura", "rubi-sama"),
    (688579, 125, 240, "kuroba_kasumi", "wori"),
    (688579, 843, 859, "kuroba_kasumi", "wori"),
    # shona mitsuishi
    (2216911, 2, 10, None, "shona mitsuishi"),
    # Gensou no Idea
    (839435, 2, 90, "nanami_naru", "makita masaki"),
    (839435, 100, 169, "nanami_naru", "makita masaki"),
    (839435, 176, 415, "shinomori_rinon", "makita masaki"),
    (839435, 402, 421, "kenzaki_noel", "makita masaki"),
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
    # Yakusoku no Natsu, Mahoroba no Yume
    (1230539, 727, 1421, "azuma_nagisa", "chikotam"),
    # Hare Nochi Kitto Nanohana Biyori
    (1919557, 3, 97, "ayasaki_nanoka", "chikotam"),
    (1919557, 98, 202, None, "chikotam"),
    (1919557, 203, 275, None, "sakana"),
    (1919557, 276, 358, None, "sakura hanpen"),
    (1919557, 396, 420, "ayasaki_nanoka", "chikotam"),

    (733798, 2, 96, "ayasaki_nanoka", "chikotam"),
    (733798, 97, 201, None, "chikotam"),
    (733798, 202, 274, None, "sakana"),
    (733798, 275, 323, None, "sakura hanpen"),
    (733798, 324, 348, "ayasaki_nanoka", "chikotam"),
    # pieces
    (1445329, 1, 556, "kimihara_yua", "mikagami mamizu"),
    (1390124, 1, 146, "kimihara_yua", "mikagami mamizu"),
    (1647868, 6, 143, "kimihara_yua", "mikagami mamizu"),
    # Unless Terminalia
    (2175956, 2, 169, "mikuriya_ren", "mikagami mamizu"),
    # shiratama
    (2616641, 40, 55, None, "shiratama"),
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
    (1149861, 768, 873, "asahina_seri", "ayuma sayu"),
    (1150058, 1, 552, "asahina_seri", "ayuma sayu"),
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

    (762285, 971, 1107, "hasugase_mina", "motoi ayumu"),
    (762285, 2, 21, "hayami_asagao", "motoi ayumu"),
    (762285, 35, 128, "hayami_asagao", "motoi ayumu"),

    (1076280, 1432, 1673, "hasugase_mina", "motoi ayumu"),
    (1076280, 38, 188, "hayami_asagao", "motoi ayumu"),
    (1076280, 647, 693, "hayami_asagao", "motoi ayumu"),

    # motoi ayumu
    (491097, 273, 876, None, "motoi ayumu"),
    (634769, 132, 276, None, "motoi ayumu"),
    (634769, 634, 881, None, "motoi ayumu"),
    (1907517, 3, 164, None, "motoi ayumu"),
    (1907517, 328, 545, None, "motoi ayumu"),
    # Koikishi Purely ☆ Kiss
    (875317, 320, 708, "shidou_mana", "yuuki hagure"),
    (875317, 881, 1273, "elcia_harvence", "yuuki hagure"),
    (875317, 1312, 1667, "fujimori_yuu", "yuuki hagure"),
    (875317, 725, 805, "bernadette_villeburg", "yuuki hagure"),

    (1302133, 1, 558, "elcia_harvence", "yuuki hagure"),
    (1302133, 559, 936, "bernadette_villeburg", "yuuki hagure"),
    (1302133, 1417, 1936, "shidou_mana", "yuuki hagure"),
    (1302134, 430, 913, "fujimori_yuu", "yuuki hagure"),
    # Juukishi Cutie ☆ Bullet
    (840881, 163, 266, "reina_de_medishi", "yuuki hagure"),
    (1868156, 756, 1100, "reina_de_medishi", "yuuki hagure"),

    # D.S. -Dal Segno
    (1083084, 2, 135, "asamiya_himari", "tanihara natsuki"),
    (1056040, 178, 281, "asamiya_himari", "tanihara natsuki"),
    (1056040, 500, 507, "asamiya_himari", "tanihara natsuki"),
    # D.C.4 ~Da Capo 4~
    (1994876, 1, 124, "sagisawa_arisu", "tanihara natsuki"),
    (1994876, 467, 585, "sagisawa_arisu", "tanihara natsuki"),
    (1994876, 652, 696, "mishima_miu", "kisaragi yuu"),
    (1994876, 201, 247, "mishima_miu", "kisaragi yuu"),
    (2205861, 1, 73, "sagisawa_arisu", "tanihara natsuki"),
    (2205861, 126, 153, "mishima_miu", "kisaragi yuu"),
    (2205861, 419, 465, "mishima_miu", "kisaragi yuu"),
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
    (1354206, 518, 794, "mihama_inori", "yuzuna hiyo"),
    (1354206, 1291, 1608, "mihama_inori", "yuzuna hiyo"),
    (1354206, 1138, 1290, "ikegai_mayu", "konomi"),
    (1354206, 1941, 2000, "ikegai_mayu", "konomi"),
    (1354273, 106, 387, "ikegai_mayu", "konomi"),
    # 12 no Tsuki no Eve
    (671506, 593, 1039, "unahara_yuki", "yuzuna hiyo"),
    (671506, 1040, 1140, "shiina_mizuka", "takasaki maco"),
    (671507, 1, 305, "shiina_mizuka", "takasaki maco"),
    (671507, 306, 530, "shiina_anzu", "shona mitsuishi"),
    (671507, 531, 677, "shiina_mizuka", "takasaki maco"),
    (671507, 678, 853, "shiina_anzu", "shona mitsuishi"),
    # Yome Sagashi ga Hakadorisugite Yabai.
    (2971469, 2, 134, "yagami_kanna", "ikegami akane"),
    (2971469, 418, 553, "takamiya_nanaka", "ikegami akane"),
    (1196589, 1, 105, "yagami_kanna", "ikegami akane"),
    (1196589, 380, 474, "takamiya_nanaka", "ikegami akane"),
    (878236, 3, 135, "yagami_kanna", "ikegami akane"),
    (878236, 432, 555, "takamiya_nanaka", "ikegami akane"),
    # Deatte 5-fun wa Ore no Mono! Jikan Teishi to Atropos
    (1305605, 171, 312, "kurose_sakura", "ikegami akane"),
    (1305605, 611, 772, "hiiragi_hakua", "ikegami akane"),
    (1375991, 145, 250, "kurose_sakura", "ikegami akane"),
    (1375991, 505, 666, "hiiragi_hakua", "ikegami akane"),
    # Natsuiro Recipe
    (819570, 157, 279, "yaehara_yuzu", "non"),
    # Hatsukoi Sankaime
    (1009352, 2, 363, None, "chikotam"),
    (1009352, 364, 813, "hikami_yurino", "narumi yuu"),
    (1009352, 814, 1238, "emiliya_karimov", "narumi yuu"),
    (1009352, 1239, 1920, None, "takashina at masato"),
    # Hakoniwa Logic
    (753681, 4, 103, "maezono_kirika", "yukie"),
    (753681, 106, 217, "iriya_koko", "yukie"),
    (753681, 218, 322, "kidou_shizuku", "miwa futaba"),
    (753681, 323, 449, "amesara_mana", "miwa futaba"),
    (753681, 450, 537, "sakuraba_moemi", "yukie"),

    (3595341, 1, 810, "maezono_kirika", "yukie"),
    (3595341, 811, 1440, "iriya_koko", "yukie"),
    (3595341, 1441, 1690, "kidou_shizuku", "miwa futaba"),
    (3595466, 511, 850, "amesara_mana", "miwa futaba"),
    (3595466, 1, 510, "sakuraba_moemi", "yukie"),
    # QUINTUPLE☆SPLASH
    (886805, 2, 79, None, "sakana"),
    (886805, 80, 216, None, "mikeou"),
    (886805, 217, 343, None, "yukie"),
    (886805, 344, 446, None, "ichiri"),
    (886805, 447, 538, None, "sakura hanpen"),

    (886805, 539, 563, None, "sakana"),
    (886805, 564, 587, None, "mikeou"),
    (886805, 588, 615, None, "yukie"),
    (886805, 616, 640, None, "ichiri"),
    (886805, 641, 737, None, "sakura hanpen"),

    (1134306, 1, 224, None, "sakana"),
    (1134306, 225, 672, None, "mikeou"),
    (1134306, 673, 1120, None, "yukie"),
    (1134306, 1121, 1582, None, "ichiri"),
    (1134306, 1583, 1822, None, "sakura hanpen"),

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

    (1093800, 3, 60, "rousen'in_rise", "yamakaze ran"),
    (1093800, 61, 118, "sagisawa_chiho", "saeki nao"),
    (1093800, 119, 174, "kitamikado_ritsuko", "saeki nao"),
    (1093800, 175, 226, "kitamikado_ayaka", "yamakaze ran"),
    (1093800, 511, 603, "myougi_marika", "yamakaze ran"),

    (1329062, 2, 59, "rousen'in_rise", "yamakaze ran"),
    (1329062, 60, 117, "sagisawa_chiho", "saeki nao"),
    (1329062, 118, 173, "kitamikado_ritsuko", "saeki nao"),
    (1329062, 174, 225, "kitamikado_ayaka", "yamakaze ran"),
    (1329062, 506, 598, "myougi_marika", "yamakaze ran"),
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
    (3019741, 1, 2000, "mikoto", "nanaca mai"),
    (3498433, 1, 77, "mikoto", "nanaca mai"),
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
    (1964980, 2, 250, "tenkawa_sayuki", ""),
    (1964980, 443, 632, "orime_tamaki", None),

    # Sennagi
    (2997104, 1, 1650, "mikosono_himeka", "chobipero"),
    (2997104, 1651, 1758, "eru", "aiu"),
    (2997102, 1, 1882, "mikosono_himeka", "chobipero"),
    (2997102, 1896, 2000, "eru", "aiu"),
    (2720188, 1, 885, "mikosono_himeka", "chobipero"),
    (2720188, 886, 1018, "eru", "aiu"),
    (2997103, 1, 57, "eru", "aiu"),

    # Haison Shoujo
    (2710683, 1, 90, "kagome", "aiu"),
    (2710683, 91, 206, "yakumo_azusa", "yuurin"),
    (2710683, 207, 272, "kagami_shuri", "chobipero"),
    (2710683, 275, 344, "emma_aaron_yakushiin", "aose"),
    (2710683, 346, 432, "karasuno_tsubame", ""),
    (2710683, 435, 518, "furube_yurara", ""),
    (2710683, 521, 589, "osakabe_rei", ""),
    (2710683, 592, 660, "yuzuriha_manaka", ""),

    (2410167, 10, 183, "kagome", "aiu"),
    (2410167, 413, 660, "yakumo_azusa", "yuurin"),
    (2410167, 184, 412, "kagami_shuri", "chobipero"),
    (2410167, 661, 770, "emma_aaron_yakushiin", "aose"),
    (2410167, 771, 881, "karasuno_tsubame", ""),
    (2410167, 882, 988, "furube_yurara", ""),
    (2410167, 989, 1105, "osakabe_rei", ""),
    (2410167, 1106, 1208, "yuzuriha_manaka", ""),

    (3103887, 1, 174, "kagome", None),
    (3103887, 175, 388, "yakumo_azusa", None),
    (3103887, 389, 614, "kagami_shuri", None),
    (3103887, 615, 761, "emma_aaron_yakushiin", None),
    (3103887, 762, 872, "karasuno_tsubame", None),
    (3103887, 873, 979, "furube_yurara", None),
    (3103887, 980, 1096, "osakabe_rei", None),
    (3103887, 1097, 1199, "yuzuriha_manaka", None),
    # Oniichan migite no shiyou wo kinshi shimasu!
    (1179865, 13, 510, None, "k-ko"),
    (1179865, 784, 804, None, "k-ko"),
    (1179865, 511, 1025, None, "hisama kumako"),

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
    (1175726, 499, 648, "kumakura_yoake", "hisama kumako"),
    (1175726, 3, 127, "hanatsuka_aika", "ameto yuki"),
    (3428412, 1, 798, "kumakura_yoake", "hisama kumako"),
    (3428411, 1, 330, "hanatsuka_aika", "ameto yuki"),
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

    # uso(campus)
    (878564, 1, 2000, "himeno_satsuki", "riichu"),
    (928204, 1, 2000, "izumi_aoi", "riichu"),
    (998646, 1, 2000, "eris_fall_cartlet", "riichu"),
    (1096716, 1, 2000, "teidou_setsuka", "riichu"),

    # Kanojo no Seiiki
    (2493223, 1, 2000, "akiyoshi_fuyuka", "ryohka"),
    (769445, 1, 2000, "akiyoshi_fuyuka", "ryohka"),
    (847646, 1, 136, "nase_yukana", "ryohka"),
    (2593173, 12, 163, "nase_yukana", "ryohka"),
    (1000114, 5, 116, "ootori_maika", "ryohka"),
    (1000114, 117, 231, "nase_yaeka", "ryohka"),
    (2771291, 5, 116, "ootori_maika", "ryohka"),
    (2771291, 117, 231, "nase_yaeka", "ryohka"),
]

# 提取目录与图片序号：.../webp/<dir>/image_<num>.webp
PATH_RE = re.compile(r"/webp/(\d+)/image_(\d+)\.webp$")

def lookup_targets(dir_id: int, num: int) -> Optional[Tuple[str, str]]:
    for d, start, end, character, artist in RANGES:
        if dir_id == d and start <= num <= end:
            return character, artist
    return None

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

            path = obj.get("path", "")
            m = PATH_RE.search(path)
            if m:
                dir_id = int(m.group(1))
                num = int(m.group(2))
                targets = lookup_targets(dir_id, num)
                if targets:
                    target_character, target_artist = targets
                    # 命中范围：覆盖 character 与 artist
                    if (target_character and obj.get("character") != target_character) or (target_artist and obj.get("artist") != target_artist):
                        obj["character"] = target_character if target_character else obj.get("character", "")
                        obj["artist"] = target_artist if target_artist else obj.get("artist", "")
                        modified += 1

            # 紧凑写回，保持一行一个 JSON
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return modified

def main():
    ap = argparse.ArgumentParser(description="Set character/artist for specific image ranges in JSONL.")
    ap.add_argument("inputs", nargs="+", help="input JSONL path pattern(s), supports * wildcard; add final path as OUTPUT when not using --inplace")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input file in place")
    args = ap.parse_args()

    patterns: List[str]
    output_path: Optional[str] = None
    if args.inplace:
        patterns = args.inputs
    else:
        if len(args.inputs) < 2:
            ap.error("非 --inplace 模式下请提供输入和输出文件，如: script in.jsonl out.jsonl")
        *patterns, output_path = args.inputs

    expanded_inputs: List[str] = []
    for pattern in patterns:
        if glob.has_magic(pattern):
            matches = sorted(glob.glob(pattern))
            if not matches:
                ap.error(f"模式 {pattern} 未匹配到任何文件。")
            expanded_inputs.extend(matches)
        else:
            if not os.path.exists(pattern):
                ap.error(f"输入文件 {pattern} 不存在。")
            expanded_inputs.append(pattern)

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
