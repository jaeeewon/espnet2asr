from .joiner_abs import AbsJoiner


class JoinerV2(AbsJoiner):
    def __init__(self, every=0.5, threshold=0.15, log=True):
        self.data: list[tuple[float, float, str, int, int]] = (
            []
        )  # [ (second, min_avg, recognized, start_w, seg_idx), ]
        self.every = every
        self.THRESHOLD = threshold
        self.log = log

    def append_context(
        self,
        start_window: int,
        start_second: float,
        end_second: float,
        recognized: str,
        min_avg: float,
        segment_index: int,
    ):
        second = start_window * self.every + start_second
        for i, (sec, mavg, rec, start_w, rel_w) in enumerate(self.data):
            if (abs(second - sec) < self.THRESHOLD) or (
                start_w == start_window and rel_w == segment_index
            ):
                if mavg > min_avg:
                    # self.data[i] = (second, mavg, rec, start_window, segment_index)
                    # self.data[i][0] = second
                    # self.data[i][3] = start_window
                    # self.data[i][4] = segment_index
                    if self.log:
                        print(
                            f"not update exist token {rec} =/> {recognized} to abs {sec:.2f}s | old_mavg {mavg:.2f} | new_mavg {min_avg:.2f} > {self.get_string()}"
                        )
                    return

                self.data[i] = (
                    second,
                    min_avg,
                    recognized,
                    start_window,
                    segment_index,
                )
                if self.log:
                    print(
                        f"update exist token {rec} > {recognized} to abs {sec:.2f}s | old_mavg {mavg:.2f} | new_mavg {min_avg:.2f} > {self.get_string()}"
                    )
                return

        # 안 될 조건
        # start_window != 0
        # and
        # segment_index == 0
        if not (start_window and not segment_index):
            self.data.append(
                (second, min_avg, recognized, start_window, segment_index)
            )  # THRESHOLD 안에서 겹치는 게 없다면 추가
            if self.log:
                print(
                    f"inserted new token {recognized} newly | new_mavg {min_avg:.2f} > {self.get_string()}"
                )

    def get_string(self):
        tokens = [a[2] for a in sorted(self.data, key=lambda x: x[0])]
        return " ".join(tokens)
