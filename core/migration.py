import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .json_store import (
    JsonStoreError,
    read_json_object,
    update_json_object,
    write_json_object,
)
from .logger import log
from .theme_defaults import extract_theme_defaults

_LOG_PREFIX = "Defaults"
_MANIFEST_NAME = ".manifest.json"
_DEFAULT_VALUE_UPGRADES = {
    Path("config/system_prompts.json"): {
        "Scene Motion Analysis": (
            "2d19990115e48056b4c03bcad313bf29905a1a081bebdeaa0316047d2b73e724",
            "7c79a5b1737f0da4534f7e73ad0393825db7527045f39e911cc02cd4d5080197",
            "37c6e3222199c1bfdabd741cbe0d20c6d00bf052eba5850c20823542ee753e7c",
            "fd31bf57908fa3c5ae71737962a29a2032266a860eaa5c603c3c24557764201b",
            "4eaa0d61488fb8b2ccabc649aeac083b79072fb6494228b9029e6a621f2d1d1f",
            "1ddd6fd73194869e50d35bac16436a52028e59d53e98547b5ea6506e3c972115",
            "fe779c2bf14f1dd2523ebd6ac0ece1ccee455fa6bfe767d4bfc53704bcaa1d35",
            "e3c9d8f7719220558da3888b3c3cb0af1c18e18f1a058b148bc3885233f46d78",
        ),
        "Scene Description (T2V)": (
            "e5230e5988a684dc9b82137d241b932d0d723ac8b9126d8f32dafb6173f71f4e",
        ),
        "Simple Description": (
            "041c44d1996e1c159dcc9ea5151d0bded5a1c752ed3427a3de088bc34ed3adc0",
            "6c9dbece46a6351d5ae1b5f1b0b425fe00be57faf2ed2b208c6a9b729b92c348",
            "b8743371674846abe85fcfa070a9c2ab37e9eedd52a09e46dff6f8f09e14f8f8",
            "aeefd33e7117ec9e48b994728a1929351130343335a1f2c76739130120982206",
        ),
        "Detailed Description": (
            "82a43e6b8c0d2e81d249b8109c1d1f9a462d57e5ba474efe2e269600f0c9ae4c",
            "539cd7b53e972bb41bdc3af3a4b4d3327a31742a5e9af93328620702e4f94f60",
            "bfa6653d8db7a7a6740adef5cb676fe6608ebef31150a29646a4beb706aea326",
            "a081bc8062e696681d78d9ff99b939b1eb81d768285b0cf530c89e0296c011ac",
        ),
        "Ultra Detailed Description": (
            "707b8e6f4a12797c35a6a92eae6caafa9ad20c21904e9af605a611553622efc3",
            "966d72276cd282a303ec7ffb59f3f6243d0d93c84cfaf27bdb107562ca0084e8",
            "ac937870c4ce4a36366800ec04aa83f0f837c2a3fab4fec9123138a894409a3c",
            "71a5334a06d9201ef92defdb94aa58d57bea81169c4a7385753087cce05b898e",
            "fb439f50c83c5210896e0b146b37f8d3de041cded00c4605570f31aa224e56fa",
        ),
        "MiniMax Music 3": (
            "6db71d6f7fd9890a6f75127e1e290cbe72988d5b6d39d99812ee4a3cd738e739",
            "aba0293889df90a05a4a2f14c083ec10d22b75f37b3f2fa3a173dc0bd2253701",
        ),
        "MiniMax H3 Scene 5s": (
            "e03e44bae85d73581a1908c7dc21e0c898a008019096482631d743b09794ffe8",
            "0b7efa02451ded08444004329cfc3defa0b5fd9b558dd5c5493e068643d0caaa",
            "3b64d6cbbbde03aec3da19823c6d7ac229f079ccc7fc1c4f7d33012eef57b4f1",
        ),
        "MiniMax H3 T2VA Timeline 15s": (
            "4088c51d1f3c1aa8ca8d55ff67302e4b1ddd57abdd545961f64619d640d09a16",
        ),
        "MiniMax H3 I2VA Timeline 15s": (
            "a0db383415417dc60a310a0cf6bc7414af6502b8aa278bd63ff71c7e7685fe56",
        ),
        "MiniMax H3 FL2VA Timeline 15s": (
            "2bcd528f236f2ef64ee26a75cfeff64080a9e891b2ba5333b32e7a652f784a62",
        ),
        "MiniMax H3 L2VA Timeline 15s": (
            "118548380acf93fe8e2352246538b57ce1560ce4a835ca9b218dd33433899ce2",
        ),
        "Wan 2.2 Scene 5s": (
            "4e85319ae508d07c8df224dabb11c02cd686f88ef11557071d1d7c7aae48ff5f",
            "0a59a674f94312cc2dedab181daf52eba290c756c2b914f2a7816e3353f52d6e",
        ),
        "Wan 2.2 Timeline 5s": (
            "955c2d2efd52d093fa57ff646652615a145b0e7b46d60f95909af2c52987d5e0",
            "84c9716097afda70e1a7469fd156b7f339caa1709f6bef4f7cea9785ed34d401",
        ),
        "Wan 2.2 Timeline 5s 2s": (
            "88d1e095216abcd1acd1721f9b63f4e104f93711e261f5b7553b6f9b15d2c0d0",
            "ec85b22d4f60b51b9479502f4022101c6fcc981f14e74dcd1a6a9adb602bb1ef",
        ),
        "Wan 2.2 Timeline 5s 3s": (
            "46761f8c47641b6d776a8b1d3ba29806f26aa48ad904b9fa3c10b2076b847225",
            "06d1b671360c1e2e903766e299570d42a9e7cd2a90cc51c05d21690a417c6aa4",
        ),
        "Wan 2.2 Scene 10s": (
            "ecd4a75b5b62b9028fda5e4f23a23bad082620f61e99266f67d270fd62d59513",
            "e8f095f1e40a010c58dea429c933a13f7d921c5139065334b49210d149b63cd2",
        ),
        "Wan 2.2 Timeline 10s": (
            "361d1dcd651abaf2cbbdf07224d01e96d2c3e06704dbc928bd492e895e1ab722",
            "6ee457b9605c7c1508726a7cb113ad5bd24e5cb838a8c1a4424b97712d1c335a",
        ),
        "Wan 2.2 Scene 20s": (
            "7a919e2e6e6b286818a2cc640bac7fb580f66dd09a2628b9b9f95dd9f4ef05e2",
            "d7254828b56fb881a36bd56afc63f7f86654dfd4c81fc3529e87ada1be5ed99b",
        ),
        "Wan 2.2 Timeline 20s": (
            "706468776084156d1c42bbbfcbb7556b47309740dafb1561ed3b133d47a40b02",
            "266d493f848267eb0cbffb3636bc5c6d3aea594b27088db750252f5923932d89",
        ),
        "Wan 2.2 CN Atomic": "7a5523da49285d67ce93f538a6b89c538ee4ab8e65900805e4b28a11321220cd",
        "LTX 2.3 I2V": "dbf829e66bc2232fb5dd542c069f804b81641e2add4cc3233c9d8def4b705eeb",
        "Cinematic Description": (
            "9c987ea8d1c83535f515ed125767455d84839235b58acaf934c39a02ec5376b8",
        ),
        "Image Analysis": (
            "74ab8e9bb7111cf6622365620153a31315c08c8517f1a6689e9ac2b0723e92fd",
        ),
        "Detailed Analysis": (
            "95ae78bea8fb2f0dad98ea3383d90472c6f9bf25ebb38f2b781947bc8be549a2",
        ),
        "Tags": (
            "77cb4e2939eba82aeaa6b66adb39b432d25534eb1827cd45446482334a174e35",
        ),
        "Video Summary": (
            "43fc1d75e8213d1ca9f8ec2085eaf833267105a933c31ef1a57ba0cec18dfb68",
        ),
        "Expand Text": (
            "6199c90b029ca5729b061301d273c0c9d7dafbcecee3452477e46f5fffe3a913",
        ),
        "Refine & Expand Prompt": (
            "cfeba7cfa367598853a95b5afca146e30e08538a5006a23969182051e9177885",
        ),
        "Rewrite Style": (
            "32635e42d2fad2997220452a1ebd43582a030dd05b5cb2f0e8b1c0428d67c8f4",
        ),
        "Tags to Natural Language": (
            "2c2ebfb13d5f994efc3f83453466c18997a9c97d5f1fa268cf6a89e05a55b8da",
        ),
        "Natural Language to Tags": (
            "5e694c8aea7f233a18d25de5653f98e707c29587d681f882a18f80cc8da05840",
        ),
        "Translate to English": (
            "1cf514477ea1e3150ce3ccd84ced9180c77044c808830f67d03009f81518780e",
        ),
        "Short Story": (
            "2f4f57e30e94e5252ab892a0cd124993913c64a45bf7e329bd76a30f996e8394",
        ),
        "Summarize": (
            "a0c4e830179f88f0df3c232e70ebfc39079d350e35b86d0a9d14da75f0a9151d",
        ),
        "Prompt Variations": (
            "21fa83e7d9c9b9eee011a813fb6f94d06272fcdbd735f70a89a8d362b42110c4",
        ),
    },
    Path("config/system_prompts_nsfw.json"): {
        "MiniMax H3 Scene": (
            "0dc68da3d35ad2bf333918a87f975b10cbfa3cd561a19f2dac0694836090ec7f",
        ),
        "MiniMax H3 T2VA Timeline": (
            "e73f1481789e886857611e630339c6257d91fd047ffa15ea81c7a3fb023d0f23",
        ),
        "MiniMax H3 I2VA Timeline": (
            "4777b732afb13c79fd7c31894dfd16cd3676f9d3b40a9da1dd63888cbe7307b0",
        ),
        "MiniMax H3 FL2VA Timeline": (
            "8c953cba33cdce074ba52bd8a2a9d76b10fd40736309726d424ad1acd890374e",
        ),
        "MiniMax H3 L2VA Timeline": (
            "46635e63f79cbb235782b7def702f39c978d9d627ee348f5e22b81f89b744880",
        ),
        "Scene Motion Analysis": (
            "2d19990115e48056b4c03bcad313bf29905a1a081bebdeaa0316047d2b73e724",
            "7c79a5b1737f0da4534f7e73ad0393825db7527045f39e911cc02cd4d5080197",
            "37c6e3222199c1bfdabd741cbe0d20c6d00bf052eba5850c20823542ee753e7c",
            "fd31bf57908fa3c5ae71737962a29a2032266a860eaa5c603c3c24557764201b",
            "4eaa0d61488fb8b2ccabc649aeac083b79072fb6494228b9029e6a621f2d1d1f",
            "1ddd6fd73194869e50d35bac16436a52028e59d53e98547b5ea6506e3c972115",
            "fe779c2bf14f1dd2523ebd6ac0ece1ccee455fa6bfe767d4bfc53704bcaa1d35",
            "e3c9d8f7719220558da3888b3c3cb0af1c18e18f1a058b148bc3885233f46d78",
        ),
        "Scene Description (T2V)": (
            "e5230e5988a684dc9b82137d241b932d0d723ac8b9126d8f32dafb6173f71f4e",
        ),
        "Simple Description": (
            "b8743371674846abe85fcfa070a9c2ab37e9eedd52a09e46dff6f8f09e14f8f8",
        ),
        "Detailed Description": (
            "bfa6653d8db7a7a6740adef5cb676fe6608ebef31150a29646a4beb706aea326",
        ),
        "Ultra Detailed Description": (
            "ac937870c4ce4a36366800ec04aa83f0f837c2a3fab4fec9123138a894409a3c",
            "b8d052ea0e0fcfec00f4114d166bc46e38c94c4482125255438c8007291fba2a",
        ),
    },
    Path("config/llm_few_shot_training.json"): {
        "simple_description": (
            "f38b74170016abe20aa34caa041f647f6519f3725b6d63bbc1fd0c1c9279b55b",
            "c9117e9d305351199e3bb2513fae0cbd05cb37beacbda21fcbea41da3c6b7249",
            "f4172d60a24d623393720fa758a95bef9d0afa4c19066f874e1183dfd61e37a1",
        ),
        "detailed_description": (
            "8d5d08dd2a4a59dde5d9205fb9195e4f37714989078fa486b5b7d09c7a0ba541",
            "33c948b5a3e059dcc48271921362a8bca3d48cb93cb71032cb5c792275ec4e4e",
            "7c3aa418fd5d0ed58c0a6f128cca406082e779762521eaa60daa478b4daa587f",
        ),
        "ultra_detailed_description": (
            "fb60bbcb8d2126423a055fa1addf16204775f4948d0a28a0c98fd1066e2bd3c7",
            "0ba3a53f902a95506c4a9350945e4523efb2072e331ad2d4f89762e169b85fea",
            "202c487116e05b4e6cef4d6e4293630c558ba3edb4cb80ad1b864e2d566933f5",
            "79430a0b0be51b537aadaef371e87bb25573e90389808b2a6f3dbe74754ca79b",
        ),
        "minimax_music_3": (
            "eb4dd0d3053de1b777d09b24fea6ef467976fbe0a3c0c5215e94705c4364cc16",
            "de9b9b0554e455f994fbc2160d9b4b87f1f21764d19dbf65c18985449ebd7919",
        ),
        "minimax_h3_scene_5s": (
            "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
            "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
            "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
        ),
        "minimax_h3_i2va_timeline_15s": (
            "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
        ),
        "minimax_h3_fl2va_timeline_15s": (
            "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
        ),
        "minimax_h3_l2va_timeline_15s": (
            "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
        ),
        "wan_2.2_scene_5s": "bd734ddd2927506eea6208fc2eaf9f3aac38601853fac3e360c439b43a6b4712",
        "wan_2.2_timeline_5s": "74ce9c85afcb14c01dcc48ce682c8cd844aabd7cedb14c3e4c5644383df2579c",
        "wan_2.2_timeline_5s_2s": "8c63941e5b244b21a6b5473afea1e2a7c438a6fbf6114f488df4fd36136166a8",
        "wan_2.2_timeline_5s_3s": "dbfc717968a30a7edfe0cf1461708a528bc30e295acf1bdf8e0be757266faf09",
        "wan_2.2_scene_10s": "289bd17589b10f0eb2bcab9b8a9e42fd37bc2d30c1884ef5ed4aa08f43b707ea",
        "wan_2.2_timeline_10s": "839803e40b31b23d11f6e234a8788a3893d87ff8a2620c7b05110a182d178714",
        "wan_2.2_scene_20s": "cfde6cefcb4e1e1af9c06004dfe5d877e75b41dde6bf8f63d395bd5db666d8c5",
        "wan_2.2_timeline_20s": "3cf9e9399d9f94bbcfe34cb8d85ddc12dc0ea931c98f5d75a1f6ae918713e53b",
        "ltx_2.3_i2v": "ddddcb90ebc4292ae937be87f8fe1d542e6ab55416fc777a56dd44208ea4a397",
    },
    Path("config/llm_few_shot_training_nsfw.json"): {
        "minimax_h3_t2va_timeline": (
            "8765d411ab0a4e3206cf3d2ea8d82784b1f656eaf67332f81c2efc9628d4a9ca",
        ),
        "simple_description": (
            "7d9c53aff8e0665d301dcc9828dabf2d28e2a5d90e0eeea346ca32548400f6ba",
            "e01a31c550c8037295eefa485c80ad28beaea429a352d5ee801d09961828a312",
            "1475753780fd1b58be1730186fa5372fdc128f29242faa8cdc9f01cc002480b2",
            "e7fde3e7566234c3ffa5bf16893548f673ce16399ea5445dea3264318423bbb2",
        ),
        "detailed_description": (
            "473142323f9a62e263cd2beffad5d6030bfb65f1fcfce95ad643df2e3da82f9a",
            "b2f5a72fa125d6c305b8190e88af80ef6278369fde41080b31c4c69f184e3648",
            "88b3739c3171319ff715a648bcbf039b421db1a05ce8dc70a3737a76572f6b73",
            "2ba8a18b1bb535ea244de4795af7b2e813002dbbd7c786e33b385e8eb31107f9",
        ),
        "ultra_detailed_description": (
            "99438a74d7201070b5fea1a70c1294071864751b77664c8fb77c56d3014c5b8b",
            "c43625ab470e6a5d0484b95e670c44e4ffe661ff3280a5ad7f6ae5320d33e156",
            "6142924736694c7a4b93f7d13a5a9254c147414c5302a908ff7af65d5876cffd",
            "a7ae88ba1882da33507da16f01fb8cbeff21cb0310ed9dfa9f1413d204c92e21",
            "ba335eab8104debd7727570f1e716b2577088f925d8cc8140573d1b47b74b565",
        ),
        "minimax_music_3": (
            "eb4dd0d3053de1b777d09b24fea6ef467976fbe0a3c0c5215e94705c4364cc16",
            "de9b9b0554e455f994fbc2160d9b4b87f1f21764d19dbf65c18985449ebd7919",
        ),
        "minimax_h3_scene_5s": (
            "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
            "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
            "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
        ),
        "minimax_h3_i2va_timeline_15s": (
            "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
        ),
        "minimax_h3_fl2va_timeline_15s": (
            "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
        ),
        "minimax_h3_l2va_timeline_15s": (
            "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
        ),
        "wan_2.2_scene_5s": "1c61cde6ddc8c83ef910f2bee4fa6ce644cecd6f300a76267ba000e75e793f44",
        "wan_2.2_timeline_5s": "ce2153d353b6849013343f621fd7beb47a77a0ca32eff391eb5a97905a1b10b9",
        "wan_2.2_timeline_5s_2s": "8c63941e5b244b21a6b5473afea1e2a7c438a6fbf6114f488df4fd36136166a8",
        "wan_2.2_timeline_5s_3s": "dbfc717968a30a7edfe0cf1461708a528bc30e295acf1bdf8e0be757266faf09",
        "wan_2.2_scene_10s": "1fdf84cd819dc66ca8f23fd7fc2c5e4dd452b2410e60513632888792abe5ef3f",
        "wan_2.2_timeline_10s": "95989ed6d728aede0e8e811eb4e420e2f523f6a35e79a2155dbd87323bc4dad4",
        "wan_2.2_scene_20s": "cfde6cefcb4e1e1af9c06004dfe5d877e75b41dde6bf8f63d395bd5db666d8c5",
        "wan_2.2_timeline_20s": "3cf9e9399d9f94bbcfe34cb8d85ddc12dc0ea931c98f5d75a1f6ae918713e53b",
        "ltx_2.3_i2v": "54d0a9b12e92b6eace281b066c6758d8ddc89fdb54479a215d5fa075c9e4dbbb",
    },
}

_DEFAULT_KEY_RENAMES = {
    Path("config/system_prompts.json"): {
        "MiniMax H3 Scene": (
            "MiniMax H3 Scene 5s",
            (
                "e03e44bae85d73581a1908c7dc21e0c898a008019096482631d743b09794ffe8",
                "0b7efa02451ded08444004329cfc3defa0b5fd9b558dd5c5493e068643d0caaa",
                "3b64d6cbbbde03aec3da19823c6d7ac229f079ccc7fc1c4f7d33012eef57b4f1",
                "b183658396a11a1fd5b945dec4ee5593b2466ba63cfc5664813bf6d319ef5553",
            ),
        ),
        "MiniMax H3 T2VA Timeline": (
            "MiniMax H3 T2VA Timeline 15s",
            (
                "4088c51d1f3c1aa8ca8d55ff67302e4b1ddd57abdd545961f64619d640d09a16",
                "e73f1481789e886857611e630339c6257d91fd047ffa15ea81c7a3fb023d0f23",
            ),
        ),
        "MiniMax H3 I2VA Timeline": (
            "MiniMax H3 I2VA Timeline 15s",
            (
                "a0db383415417dc60a310a0cf6bc7414af6502b8aa278bd63ff71c7e7685fe56",
                "4777b732afb13c79fd7c31894dfd16cd3676f9d3b40a9da1dd63888cbe7307b0",
            ),
        ),
        "MiniMax H3 FL2VA Timeline": (
            "MiniMax H3 FL2VA Timeline 15s",
            (
                "2bcd528f236f2ef64ee26a75cfeff64080a9e891b2ba5333b32e7a652f784a62",
                "8c953cba33cdce074ba52bd8a2a9d76b10fd40736309726d424ad1acd890374e",
            ),
        ),
        "MiniMax H3 L2VA Timeline": (
            "MiniMax H3 L2VA Timeline 15s",
            (
                "118548380acf93fe8e2352246538b57ce1560ce4a835ca9b218dd33433899ce2",
                "46635e63f79cbb235782b7def702f39c978d9d627ee348f5e22b81f89b744880",
            ),
        ),
    },
    Path("config/llm_few_shot_training.json"): {
        "minimax_h3_scene": (
            "minimax_h3_scene_5s",
            (
                "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
                "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
                "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
                "7096773b3774a2b94aa33d22251f7e29b147e7080a4f95806150f87f22c5ef2c",
            ),
        ),
        "minimax_h3_t2va_timeline": (
            "minimax_h3_t2va_timeline_15s",
            ("41216e089117981b5ad06827042880f29c6c1aa7129f43ce9c8625871ca20ca6",),
        ),
        "minimax_h3_i2va_timeline": (
            "minimax_h3_i2va_timeline_15s",
            (
                "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
                "89317f62af3daa87a8b239ac111d00d4296568543c776abc4a79c84c6454ec64",
            ),
        ),
        "minimax_h3_fl2va_timeline": (
            "minimax_h3_fl2va_timeline_15s",
            (
                "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
                "ff00012d89f2593f378c54516d19cde408c1a9037d7b7d281dae7ff4243601df",
            ),
        ),
        "minimax_h3_l2va_timeline": (
            "minimax_h3_l2va_timeline_15s",
            (
                "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
                "634a3c7ffd56365a31852a4241bcba52bc541b34d8135374280fa11af01a4998",
            ),
        ),
    },
    Path("config/llm_few_shot_training_nsfw.json"): {
        "minimax_h3_scene": (
            "minimax_h3_scene_5s",
            (
                "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
                "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
                "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
                "7096773b3774a2b94aa33d22251f7e29b147e7080a4f95806150f87f22c5ef2c",
            ),
        ),
        "minimax_h3_t2va_timeline": (
            "minimax_h3_t2va_timeline_15s",
            ("41216e089117981b5ad06827042880f29c6c1aa7129f43ce9c8625871ca20ca6",),
        ),
        "minimax_h3_i2va_timeline": (
            "minimax_h3_i2va_timeline_15s",
            (
                "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
                "89317f62af3daa87a8b239ac111d00d4296568543c776abc4a79c84c6454ec64",
            ),
        ),
        "minimax_h3_fl2va_timeline": (
            "minimax_h3_fl2va_timeline_15s",
            (
                "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
                "ff00012d89f2593f378c54516d19cde408c1a9037d7b7d281dae7ff4243601df",
            ),
        ),
        "minimax_h3_l2va_timeline": (
            "minimax_h3_l2va_timeline_15s",
            (
                "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
                "634a3c7ffd56365a31852a4241bcba52bc541b34d8135374280fa11af01a4998",
            ),
        ),
    },
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json_object(path)
    except (JsonStoreError, OSError) as error:
        log.warning(_LOG_PREFIX, f"Skipped unreadable defaults manifest '{path}': {error}")
        return None


def _read_bundled_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        log.warning(_LOG_PREFIX, f"Skipped unreadable bundled JSON '{path}': {error}")
        return None
    if not isinstance(data, dict):
        log.warning(_LOG_PREFIX, f"Skipped non-object bundled JSON '{path}'")
        return None
    return data


def _merge_missing(target: dict[str, Any], additions: dict[str, Any]) -> None:
    for key, value in additions.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_missing(target[key], value)


def _merge_bundled_update(
    relative: Path,
    current: dict[str, Any],
    bundled: dict[str, Any],
) -> None:
    # New defaults remain additive. Prompt entries are upgraded individually
    # only while they still match a recognized prior bundled value; local edits
    # and retired keys remain untouched.
    renames = _DEFAULT_KEY_RENAMES.get(relative, {})
    for new_key, (old_key, legacy_hashes) in renames.items():
        if new_key in current or new_key not in bundled or old_key not in current:
            continue
        if _value_hash(current[old_key]) in legacy_hashes:
            current[new_key] = copy.deepcopy(bundled[new_key])
        else:
            current[new_key] = copy.deepcopy(current[old_key])

    upgrades = _DEFAULT_VALUE_UPGRADES.get(relative, {})
    for key, legacy_hashes in upgrades.items():
        recognized_hashes = (
            (legacy_hashes,) if isinstance(legacy_hashes, str) else legacy_hashes
        )
        if (
            key in current
            and key in bundled
            and _value_hash(current[key]) in recognized_hashes
        ):
            current[key] = copy.deepcopy(bundled[key])
    _merge_missing(current, bundled)


def _load_manifest(defaults: Path) -> dict[str, Any]:
    manifest = _read_optional_object(defaults / _MANIFEST_NAME)
    return manifest if manifest is not None else {}


def materialize_defaults(repo_root: Path) -> None:
    extract_theme_defaults(repo_root)
    defaults = repo_root / ".defaults"
    manifest_path = defaults / _MANIFEST_NAME
    manifest = _load_manifest(defaults)
    changed = False
    extracted = 0
    updated = 0

    for example in sorted(defaults.rglob("*.example")):
        relative = example.relative_to(defaults)
        relative = Path(str(relative)[: -len(".example")])
        key = relative.as_posix()
        target = repo_root / relative
        example_hash = _file_hash(example)
        previous_hash = manifest.get(key)

        if relative.parts[0] == "themes":
            # Theme updates never modify an extracted user file or restore one
            # the user deleted. Keep only the shipped hash in this manifest.
            if previous_hash != example_hash:
                manifest[key] = example_hash
                changed = True
            continue

        if not target.exists():
            if previous_hash is not None:
                continue
            data = _read_bundled_object(example)
            if data is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json_object(target, data, private=relative == Path("config.json"))
            manifest[key] = example_hash
            changed = True
            extracted += 1
        elif previous_hash is None:
            data = _read_bundled_object(example)
            if data is not None:
                update_json_object(
                    target,
                    lambda current, bundled_data=data, rel=relative: (
                        _merge_bundled_update(rel, current, bundled_data)
                    ),
                    private=relative == Path("config.json"),
                )
            manifest[key] = example_hash
            changed = True
        elif previous_hash != example_hash:
            if _file_hash(target) == previous_hash:
                data = _read_bundled_object(example)
                if data is not None:
                    # A saved preference still belongs to the user when the
                    # complete config happens to match the previous defaults.
                    if relative == Path("config.json") or relative in _DEFAULT_VALUE_UPGRADES:
                        update_json_object(
                            target,
                            lambda current, bundled_data=data, rel=relative: (
                                _merge_bundled_update(rel, current, bundled_data)
                            ),
                            private=relative == Path("config.json"),
                        )
                    else:
                        write_json_object(
                            target,
                            data,
                            private=relative == Path("config.json"),
                        )
                    updated += 1
            else:
                data = _read_bundled_object(example)
                if data is not None:
                    update_json_object(
                        target,
                        lambda current, bundled_data=data, rel=relative: (
                            _merge_bundled_update(rel, current, bundled_data)
                        ),
                        private=relative == Path("config.json"),
                    )
            manifest[key] = example_hash
            changed = True

        elif relative in _DEFAULT_VALUE_UPGRADES:
            # A manifest records the bundled file, not the presence of each task
            # in runtime data. Repair missing task defaults even when hashes match.
            # The atomic updater leaves customized values and unchanged files alone.
            data = _read_bundled_object(example)
            if data is not None:
                update_json_object(
                    target,
                    lambda current, bundled_data=data, rel=relative: (
                        _merge_bundled_update(rel, current, bundled_data)
                    ),
                    private=False,
                )

    if changed:
        write_json_object(manifest_path, manifest, private=True)
    if extracted:
        log.msg(_LOG_PREFIX, f"Extracted {extracted} bundled default file(s)")
    if updated:
        log.msg(_LOG_PREFIX, f"Updated {updated} unmodified default file(s)")
