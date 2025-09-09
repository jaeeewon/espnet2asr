from realtime_joiner import RealtimeJoiner

total = ""
joiner = RealtimeJoiner()

while True:
    hyp = input("모델 추론 결과를 입력 > ")
    joiner.process_text(hyp, 0)
    if not total:
        # n줄 삭제 및 다시
        print("\033[3A", end="")

    print(f"\r\033[K실시간 대시보드")
    print(f"\r\033[K[recognized] {hyp}")
    print(f"\r\033[K[in totally] {joiner.get_text()}")
