import os
import json

work_dir = os.path.dirname(os.path.abspath(__file__))

def check_words_Chinese(words):
    """
    Check if any Chinese toxic words are present in the given list of words.

    Args:
        words (list): A list of words to check for toxic Chinese words.

    Returns:
        bool: True if any toxic Chinese word is found, False otherwise.
    """
    toxic_files = [os.path.join(work_dir, 'badwords', '广告.txt'), os.path.join(work_dir, 'badwords', '色情类.txt'), os.path.join(work_dir, 'badwords', '涉枪涉爆违法信息关键词.txt'), os.path.join(work_dir, 'badwords', '网址.txt'), os.path.join(work_dir, 'badwords', '政治类.txt')]
    #os.path.join能匹配适合系统的路径分隔符
    toxic_words = []
    #创建字典
    for file in toxic_files:
        with open(file, 'r', encoding='utf-8') as f:
            toxic_words.extend(f.read().split('\n'))
    #写入字典
    for word in toxic_words:
        if word in words and word != '':
            print('Bad cn word matched:', word)
            return True
    return False

def check_words_English(words):
    toxic_files = os.path.join(work_dir, 'badwords', 'words.json')
    with open(toxic_files, 'r') as file:
        toxic_words = json.load(file)
    for word in toxic_words:
        if word.lower() in words.lower():
            print('Bad en word matched:', words)
            return True
    return False

def main(words):
    if check_words_Chinese(words) or check_words_English(words):
        return True
