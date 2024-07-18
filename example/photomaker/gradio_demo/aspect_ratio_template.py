# Note: Since output width & height need to be divisible by 8, the w & h -values do
#       not exactly match the stated aspect ratios... but they are "close enough":)

aspect_ratio_list = [
    {
        "name": "Instagram (1:1)",
        "w": 512,
        "h": 512,
    },
    {
        "name": "35mm film / Landscape (3:2)",
        "w": 512,
        "h": 340,
    },
    {
        "name": "35mm film / Portrait (2:3)",
        "w": 340,
        "h": 512,
    },
    {
        "name": "CRT Monitor / Landscape (4:3)",
        "w": 512,
        "h": 384,
    },
    {
        "name": "CRT Monitor / Portrait (3:4)",
        "w": 384,
        "h": 512,
    },
    {
        "name": "Widescreen TV / Landscape (16:9)",
        "w": 512,
        "h": 288,
    },
    {
        "name": "Widescreen TV / Portrait (9:16)",
        "w": 288,
        "h": 512,
    },
    {
        "name": "Widescreen Monitor / Landscape (16:10)",
        "w": 512,
        "h": 320,
    },
    {
        "name": "Widescreen Monitor / Portrait (10:16)",
        "w": 320,
        "h": 512,
    },
    {
        "name": "Cinemascope (2.39:1)",
        "w": 512,
        "h": 212,
    },
    {
        "name": "Widescreen Movie (1.85:1)",
        "w": 512,
        "h": 276,
    },
    {
        "name": "Academy Movie (1.37:1)",
        "w": 512,
        "h": 372,
    },
    {
        "name": "Sheet-print (A-series) / Landscape (297:210)",
        "w": 512,
        "h": 360,
    },
    {
        "name": "Sheet-print (A-series) / Portrait (210:297)",
        "w": 360,
        "h": 512,
    },
]


aspect_ratios = {k["name"]: (k["w"], k["h"]) for k in aspect_ratio_list}
