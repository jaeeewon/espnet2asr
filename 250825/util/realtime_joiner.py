import difflib


class RealtimeJoiner:
    def __init__(
        self,
        quiet_threshold=2,
        discrete_threshold=1,
        min_speech_tokens=1,
        min_overlap_tokens=1,
        min_ratio=0.85,
    ):
        self.quiet_threshold = quiet_threshold
        self.discrete_threshold = discrete_threshold
        self.min_speech_tokens = min_speech_tokens
        self.min_overlap_tokens = min_overlap_tokens
        self.min_ratio = min_ratio

        self.buffer = ""
        self.final = []
        self.quiet_counter = 0
        self.discrete_counter = 0

    def match_overlap(self, old_text: str, new_text: str) -> str | None:
        """gen by gemini"""
        old_words = old_text.split()
        new_words = new_text.split()

        best_match_size = 0

        s = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)

        # get_matching_blocks는 (old_words 인덱스, new_words 인덱스, 길이) 튜플 리스트를 반환합니다.
        for match in s.get_matching_blocks():
            # 겹침 조건 확인:
            # 1. old_text의 끝부분과 일치하는 블록인가? (match.a + match.size == len(old_words))
            # 2. new_text의 시작부분과 일치하는 블록인가? (match.b == 0)
            # 3. 최소 겹침 단어 수를 만족하는가?
            if (
                match.a + match.size == len(old_words)
                and match.b == 0
                and match.size >= self.min_overlap_tokens
            ):

                # 겹치는 부분의 단어들을 가져옵니다.
                overlap_old_words = old_words[match.a : match.a + match.size]
                overlap_new_words = new_words[match.b : match.b + match.size]

                # 해당 블록의 유사도를 다시 한번 정확히 계산하여 임계값을 넘는지 확인합니다.
                # (autojunk=False로 인해 이미 정확하지만, 명시적으로 확인)
                checker = difflib.SequenceMatcher(
                    None, overlap_old_words, overlap_new_words
                )
                if checker.ratio() >= self.min_ratio and match.size > best_match_size:
                    best_match_size = match.size

        if best_match_size > 0:
            # 최적의 겹침을 찾았으므로, new_text의 겹치지 않는 뒷부분을 이어 붙입니다.
            new_part = " ".join(new_words[best_match_size:])
            return (old_text + " " + new_part).strip()

        # 유의미한 겹침을 찾지 못하면 None 반환
        return None

    def process_text(self, text: str, avg_db: float):
        hypo = text.strip().upper()

        if avg_db < -40 or len(hypo.split()) < self.min_speech_tokens:
            # 평균 데시벨이 일정 이하거나 최소 토큰 개수보다 작을 때
            self.quiet_counter += 1
        elif not self.buffer:
            # 첫 입력일 때
            self.buffer = hypo
            self.quiet_counter = 0
            self.discrete_counter = 0
        else:
            # 버퍼에 추가해야 할 때
            if hypo.startswith(self.buffer):
                # 버퍼만큼 그대로 인식했을 때
                self.buffer = hypo
                self.quiet_counter = 0
                self.discrete_counter = 0
            else:
                # 기존 버퍼랑 달라 비교해야 할 때(복잡한 경우)
                overlapped = self.match_overlap(self.buffer, hypo)
                if overlapped:
                    self.buffer = overlapped
                    self.quiet_counter = 0
                    self.discrete_counter = 0
                else:
                    self.discrete_counter += 1
                    if self.discrete_counter >= self.discrete_threshold:
                        self.quiet_counter = self.quiet_threshold  # 종료로 침

        self.check_quiet()
        return self.get_text()

    def check_quiet(self):
        if self.quiet_counter >= self.quiet_threshold and self.buffer:
            self.final.append(self.buffer)
            self.buffer = ""
            self.quiet_counter = 0
            self.discrete_counter = 0

    def get_text(self):
        total = [f"[{a}]" for a in self.final[:]]  # wrapped for completed
        if len(self.buffer):
            total.append(self.buffer)  # unwrapped for the other
        return " ".join(total)

    def get_text_anyway(self):
        total = self.final[:]
        if len(self.buffer):
            total.append(self.buffer)
        return " ".join(total)
