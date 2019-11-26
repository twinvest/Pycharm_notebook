from konlpy.tag import Hannanum
from collections import Counter
import pytagcloud # Add fonts supporting Korean

f = open("../crawler/description.txt", "r", encoding="UTF-8")
description = f.read()


h = Hannanum()
nouns = h.nouns(description)
count = Counter(nouns)
print(count)
tag = count.most_common(100)
tag_list = pytagcloud.make_tags(tag, maxsize=50)
pytagcloud.create_tag_image(tag_list, 'word_cloud.jpg', size=(900, 600), fontname='Korean', rectangular=False)

import webbrowser
webbrowser.open('word_cloud.jpg')