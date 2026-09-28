"""
GRID 詞彙約束校正（後處理）
句子格式固定：命令詞 顏色 介詞 數字 副詞
對每個位置的預測，找最接近的合法詞彙。
"""
import difflib

COMMANDS = ['bin', 'lay', 'place', 'set']
COLORS   = ['blue', 'green', 'red', 'white']
PREPS    = ['at', 'by', 'in', 'with']
DIGITS   = ['zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine']
ADVERBS  = ['again', 'now', 'please', 'soon']


def _closest(word: str, candidates: list) -> str:
    """從 candidates 中找最接近 word 的詞（使用序列相似度）"""
    if word in candidates:
        return word
    best = difflib.get_close_matches(word.lower(), candidates, n=1, cutoff=0.4)
    return best[0] if best else word


def correct_sentence(raw: str) -> str:
    """
    對 CTC 解碼後的句子套用 GRID 詞彙約束校正。
    輸入格式例：'bin blue with  six soon'（含空白或空格的字串）
    輸出：校正後的句子
    """
    # 切詞並過濾空字串（OOV 空白會造成多餘空格）
    words = [w for w in raw.strip().split() if w]

    if len(words) < 4:
        return raw  # 句子太短，無法判斷，保留原樣

    # 依位置做詞彙約束
    # 位置 0: 命令詞
    # 位置 1: 顏色
    # 位置 2: 介詞
    # 位置 3: 數字（GRID 格式省略了字母位）
    # 位置 4: 副詞
    corrected = []
    vocab_map = [COMMANDS, COLORS, PREPS, DIGITS, ADVERBS]

    for i, vocab in enumerate(vocab_map):
        if i < len(words):
            corrected.append(_closest(words[i], vocab))

    return ' '.join(corrected)
