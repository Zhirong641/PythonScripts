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
    (1146404, 12, 13, None, "shiramori yuse"),
    (1146404, 407, 582, None, "shiramori yuse"),
    (1146404, 891, 1017, None, "shiramori yuse"),
    (1146404, 774, 890, "ennis yutoria", "nanaroba hana"),
    (1146404, 14, 196, "konata konatsu", "kimishima ao"),
    (1409248, 1, 2000, "konata konatsu", None),
    (3000014, 35, 137, None, "kimishima ao"),
    (3000014, 220, 324, None, "shiratama"),
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
    (1245707, 13, 23, "amanogawa saya", "yashima takahiro"),
    (1245707, 48, 72, "amanogawa saya", "yashima takahiro"),
    (1245707, 107, 130, "amanogawa saya", "yashima takahiro"),
    (1245707, 305, 318, "amanogawa saya", "yashima takahiro"),
    (943537, 618, 837, "amanogawa saya", "yashima takahiro"),
    (900491, 1283, 1540, "amanogawa saya", "yashima takahiro"),
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

    (1805418, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (1805420, 1, 2000, None, "kaniya shiku, konomi, yuzuna hiyo"),
    (2313627, 1, 2000, None, "mizuno sao, satasama, yuzuna hiyo"),
    (1537715, 1, 2000, None, "yuzuna hiyo"),
    (1537491, 1, 2000, None, "yuzuna hiyo"),
    (1537567, 1, 2000, None, "yuzuna hiyo"),
    # Himawari!! -Anata Dake wo Mitsumeteru-
    (735531, 274, 380, "mikazuki tenma (himawari)", None),
    # Docchi no i ga Suki Desu ka?
    (1525889, 605, 1190, "tanemura koyuzu", "netarou"),
    # Kujiragami no Tearstilla
    (885411, 6, 98, "tenkawa mitsuki", None),
    # Floral Flowlove
    (960177, 208, 408, "mihato kano", "hontani kanae"),
    (960177, 409, 647, "tsubaki kohane", "arisue tsukasa"),
    (960177, 648, 890, "tokisaka nanao", "toranosuke"),
    # Hanayome to Maou
    (746122, 9, 67, "celica tepes lunatica", None),
    # Anata ni Koisuru Renai Recette
    (1243251, 1, 1280, "tachibana nonoka", "komeshiro kasu"),
    (1243251, 1281, 2000, "oozono yuzuki", "fumi"),
    (1243267, 1, 560, "oozono yuzuki", "fumi"),
    (1243267, 561, 1840, "kagiyoshi fuuka", "komeshiro kasu"),
    (1243267, 1841, 2000, "shirosaki mieru", "pero"),
    (1243602, 1, 1120, "shirosaki mieru", "pero"),
    (1067390, 23, 182, "tachibana nonoka", "komeshiro kasu"),
    (1067390, 183, 338, "oozono yuzuki", "fumi"),
    (1067390, 339, 498, "kagiyoshi fuuka", "komeshiro kasu"),
    (1067390, 499, 706, "shirosaki mieru", "pero"),
    # Tsumi no Hikari Rendezvous Goukaban
    (913381, 1, 248, "tsubaki fuuka", "satasama"),
    (913368, 773, 1169, "tsubaki fuuka", "satasama"),
    (913368, 1844, 2000, "tsubaki fuuka", "satasama"),
    (2657576, 774, 1170, "tsubaki fuuka", "satasama"),
    (2657600, 409, 813, "tsubaki fuuka", "satasama"),
    # Amatsutsumi
    (2285016, 334, 390, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (2285016, 972, 1130, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (1121856, 565, 1574, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
    (959791, 726, 940, "asahina kyouko (amatsutsumi)", "tsukimori hiro"),
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
    (1033787, 776, 1118, "yukishiro miu", "any, annie"),
    (1499212, 777, 1119, "yukishiro miu", "any, annie"),
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
    (1321028, 2, 112, "ootori_maria", "anapom, anapon"),
    (1321028, 230, 362, "kitazono_saya", "anapom, anapon"),
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

    # Sengokuhime 5
    (809507, 171, 210, "oda_nobuyuki_(sengoku_hime)", None),
    # amayui castle meister
    (1067242, 72, 341, "fia_(amayui_castle_meister)", "yano mitsuki"),
    (1117997, 7, 112, "fia_(amayui_castle_meister)", "yano mitsuki"),
    (1179437, 16, 35, "fia_(amayui_castle_meister)", "yano mitsuki"),
    # secret love
    (2999687, 424, 776, "akatsuka_haru", "k-ko"),
    (2999687, 1178, 1587, "natori_misa", "mango pudding"),
    (3328435, 170, 333, "akatsuka_haru", "k-ko"),
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
    (1389160, 119, 296, None, "yuzuna"),
    (1389160, 297, 434, "tomari mariko", "komeshiro kasu"),
    (1389160, 435, 586, "mizushiro_mina", "yuzuna"),
    (1389160, 591, 744, None, "anapom, anapon"),
    (1445230, 1, 400, "mizushiro_mina", "yuzuna"),
    (1445230, 401, 720, None, "yuzuna"),
    (1445230, 721, 1264, "tomari mariko", "komeshiro kasu"),
    (1445230, 1265, 1500, None, "anapom, anapon"),
    (1445272, 1, 864, None, "lucie"),
    (1445272, 865, 1080, "hino yuki", "yuzuna"),
    # Goshujin-sama, Seira ni Yume Mitai na Icha Love Gohoushi Sasete Itadakemasu ka
    (3034052, 1, 2000, "seira", "rubi-sama"),
    (2272848, 1, 2000, "seira", "rubi-sama"),
    # Wan Nyan ☆ A La Mode!
    (887743, 3, 60, " nekohana_korone", "naenae"),
    (887743, 246, 300, "inuta_hana", "naenae"),
    (1131217, 3, 81, " nekohana_korone", "naenae"),
    (1131217, 467, 555, "inuta_hana", "naenae"),
    (1886653, 2, 89, " nekohana_korone", "naenae"),
    (1886653, 407, 525, "inuta_hana", "naenae"),
    (1735897, 29, 248, " nekohana_korone", "naenae"),
    (1735897, 865, 1044, "inuta_hana", "naenae"),
    # Love Love ♥ Princess
    (839209, 3, 213, "marigold_bruette_erland", "wori"),
    (839209, 214, 233, "marigold_bruette_erland,  anastasia_imperator_erland", "wori"),
    (839209, 234, 432, "anastasia_imperator_erland", "wori"),
    (839209, 433, 592, "tsukimori_mio_erland", "wori"),
    # Love Love Life
    (688579, 2, 124, "akemiya_sakura", "rubi-sama"),
    (688579, 125, 240, "kuroba_kasumi", "wori"),
    (688579, 843, 859, "kuroba_kasumi", "wori"),
    # shona mitsuishi
    (2216911, 2, 10, None, "shona mitsuishi"),
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
