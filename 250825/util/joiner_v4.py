from .joiner_abs import AbsJoiner

# 정말 잘 나옴


class JoinerV4(AbsJoiner):
    def __init__(
        self,
        every=0.5,
        threshold=0.012,
        min_avg_threshold=0.03,
        # minimum_min_avg=-0.5,
        log=True,
    ):
        self.data: list[tuple[float, float, str, int, int]] = (
            []
        )  # [ (start_s, end_s, min_avg, recognized, start_w, seg_idx), ]
        self.every = every
        self.THRESHOLD = threshold
        self.MIN_AVG_THRESHOLD = min_avg_threshold
        # self.minimum_min_avg = minimum_min_avg
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
        min_avg = min_avg + 0.01 * segment_index
        st_sec = start_window * self.every + start_second
        ed_sec = start_window * self.every + end_second
        threshold = max(0.05, min(self.THRESHOLD * len(recognized), 0.12))
        for i, (start_s, end_s, mavg, rec, start_w, rel_w) in enumerate(self.data):
            if (
                (abs(st_sec - start_s) < threshold)
                or (abs(ed_sec - end_s) < threshold)
                or (start_w == start_window and rel_w == segment_index)
            ):
                if mavg > min_avg or abs(mavg - min_avg) < self.MIN_AVG_THRESHOLD:
                    # if abs(mavg - min_avg) < self.MIN_AVG_THRESHOLD:
                    # self.data[i] = (second, mavg, rec, start_window, segment_index)
                    # self.data[i][0] = second
                    # self.data[i][3] = start_window
                    # self.data[i][4] = segment_index
                    if self.log:
                        print(
                            f"not update exist token {rec} =/> {recognized} to abs {st_sec:.2f}s | old_mavg {mavg:.2f} | new_mavg {min_avg:.2f} > {self.get_string()}"
                        )
                    return

                self.data[i] = (
                    st_sec,
                    ed_sec,
                    min_avg,
                    recognized,
                    start_window,
                    segment_index,
                )
                if self.log:
                    print(
                        f"update exist token {rec} > {recognized} to abs {ed_sec:.2f}s | old_mavg {mavg:.2f} | new_mavg {min_avg:.2f} > {self.get_string()}"
                    )
                return

        if not (start_window and not segment_index):
            # if min_avg < self.minimum_min_avg:
            #     if self.log:
            #         print(
            #             f"pass insert new token {recognized} newly - lesser than {self.minimum_min_avg} | new_mavg {min_avg:.2f} > {self.get_string()}"
            #         )
            #     return

            self.data.append(
                (st_sec, ed_sec, min_avg, recognized, start_window, segment_index)
            )  # THRESHOLD 안에서 겹치는 게 없다면 추가
            if self.log:
                print(
                    f"inserted new token {recognized} newly | new_mavg {min_avg:.2f} > {self.get_string()}"
                )

    def get_string(self):
        tokens = [a[3] for a in sorted(self.data, key=lambda x: (x[0] + x[1]) / 2)]
        return " ".join(tokens)
