# 열람실 좌석 사람 유무 감지 — Raspberry Pi 5

카메라 모듈의 영상을 **라즈베리파이 내부**에서 처리하고, COCO 사전 학습
**YOLO11n**으로 좌석마다 사람이 있는지 확인합니다. 별도 학습이나 AI 가속기는
필수가 아닙니다. 카메라 1대로 여러 좌석을 설정할 수 있습니다.

기본 `run` 명령의 `EMPTY`는 **사람이 검출되지 않음**을 뜻합니다. 별도로 추가된
운영 코어는 사람·짐·예약·체크인을 서로 다른 정보로 저장해 무예약 점유, 노쇼,
장시간 자리 비움, 정리 요청을 구분합니다. 영상이나 얼굴 정보는 저장하지 않습니다.

## 빠른 시작

대상 환경: **Raspberry Pi 5 + Raspberry Pi OS 64-bit**. Bookworm/Python 3.11 또는
Trixie/Python 3.13의 시스템 Python을 사용합니다.
카메라 모듈은 CSI 포트에 연결합니다. Pi 5의 작은 22핀 커넥터에 맞는 케이블이
필요합니다. USB 웹캠은 `--source 0`으로 사용할 수 있습니다.

### 1. 설치

이 프로젝트 폴더를 Pi에 복사하거나, GitHub에 코드가 올라간 뒤 아래처럼 내려받습니다.

```bash
git clone https://github.com/Seojieun05/SKKU_MakerHackerton.git
cd SKKU_MakerHackerton
bash scripts/setup_pi.sh
source .venv/bin/activate
rpicam-hello -t 5000
```

설치 스크립트는 apt로 Picamera2를 설치한 뒤 `--system-site-packages` 가상환경을
만듭니다. libcamera/Picamera2는 pip로 별도 설치하지 마세요.

설치 스크립트는 OS에 설치된 NumPy 버전을 `.cache/pi-constraints.txt`에 기록하여
가상환경에서도 그 버전을 유지합니다. Picamera2/simplejpeg와 NumPy의 ABI 충돌을
줄이기 위한 설정입니다. 이후 의존성을 갱신할 때도 `-c .cache/pi-constraints.txt`를
사용하세요. 다른 OS/Python 조합은 별도 확인이 필요합니다. 실제 Pi 설치는 이 PC에서
시험하지 못했으므로 아래 검증 기록과 구분해서 보세요.

### 2. 사전 학습 모델 받기 — 인터넷은 이 단계에서 필요

```bash
python -m seat_monitor prepare
```

공식 Ultralytics 배포처에서 `models/yolo11n.pt`를 받습니다. 이미 있으면 재다운로드하지
않습니다. 이후 `run`은 모델이 없을 때 자동 다운로드하지 않고 준비 명령을 안내합니다.

### 3. 먼저 카메라 전체로 사람 탐지 확인

```bash
python -m seat_monitor run --whole-frame --preview
```

사람이 1초 이상 연속으로 검출되면 `OCCUPIED`, 8초 이상 연속으로 검출되지 않으면
`EMPTY`가 됩니다. 시작 직후와 오류 시에는 `UNKNOWN`입니다. Q 또는 Ctrl+C로 종료합니다.
`--whole-frame`은 화면 전체를 하나의 구역으로 취급하므로 복도를 지나는 사람도 포함됩니다.

### 4. 좌석 여러 개 설정

```bash
python -m seat_monitor calibrate
```

Pi에 연결한 모니터의 화면에서 좌석별 다각형을 그립니다. 캡처 한 장을 RAM에만
잠시 유지하며, 사진 파일은 저장하지 않습니다.

1. 사람이 앉았을 때 **검출 박스 중심(노란 점)**이 위치할 구역을 3~4개의 점으로 지정합니다.
   의자 바닥만 둘러싸면 앉은 사람의 상체 중심이 구역 밖에 있게 됩니다.
2. 모서리를 시계 또는 반시계 방향으로 클릭합니다. 오목하거나 교차하는 다각형은 허용하지 않습니다.
3. N: 현재 좌석 확정 후 다음 좌석. A01, A02 순서로 번호가 부여됩니다.
4. U: 마지막 점 취소. 점이 없으면 마지막 좌석을 다시 편집합니다.
5. S: 현재 좌석까지 확정하고 `config/seats.json`에 좌표만 저장합니다.
6. Q/ESC: 저장하지 않고 종료합니다.

카메라를 움직였으면 다시 설정하세요. 기존 설정을 바꾸려면 `calibrate --overwrite`를
사용합니다. 복도는 제외하고 좌석 구역끼리 겹치지 않게 그리는 것이 좋습니다.

```bash
python -m seat_monitor run --preview
```

모니터 없이 실행할 때는 미리 좌석 설정을 마친 후 `--preview`를 빼면 됩니다.

```bash
python -m seat_monitor run
```

`config/seats.example.json`은 형식 설명용 가상 좌표입니다. 실제 좌석 설정으로 그대로
사용하지 마세요. 좌표는 0~1 비율이며 해상도 변경은 허용하지만 화면 종횡비가 바뀌면
재설정을 요구합니다. 렌즈나 카메라 방향이 바뀌는 경우도 재설정이 필요합니다.

## 판정 방식과 결과

1. Picamera2가 BGR 프레임을 RAM에 제공합니다.
2. YOLO11n에서 `person` 클래스만 추출합니다. 기본 신뢰도 기준은 0.4입니다.
3. 사람 박스의 중심점이 들어가는 좌석에 배정합니다. 여러 구역에 걸리면 가장 가까운
   좌석 중심 하나에만 배정합니다. 구역 밖 사람은 좌석 판정에서 제외합니다.
4. 시간 조건을 충족한 뒤 상태를 전환합니다. 짧은 미탐지는 곧바로 빈자리로 바뀌지 않습니다.
5. `output/status.json`을 원자적으로 갱신하고 터미널에 JSON 한 줄씩 출력합니다.

`OCCUPIED` = 사람 있음, `EMPTY` = 사람 미검출, `UNKNOWN` = 아직 판단 중/확인 불가.
기본 3 FPS는 **최대 분석 빈도**이며 실측 속도 보장이 아닙니다.

```json
{
  "schema_version": 1,
  "observed_at": "2026-09-28T05:00:00+00:00",
  "valid_until": "2026-09-28T05:00:05+00:00",
  "health": "ok",
  "meaning": "person_presence_only",
  "seats": [{
    "seat_id": "A01",
    "state": "OCCUPIED",
    "has_person": true,
    "current_person_confidence": 0.9123,
    "pending_state": null,
    "state_duration_seconds": 12.3
  }]
}
```

- 시작/카메라 오류/종료 때 `UNKNOWN`으로 기록합니다. 정상 종료 후 JSON도 `UNKNOWN`입니다.
- 프로세스 강제 종료나 카메라 읽기 멈춤으로 파일이 더 이상 갱신되지 않을 수 있습니다.
  후속 웹 화면/예약 시스템은 **`valid_until`이 지났거나 `health != "ok"`이면 확인 불가**로 처리하세요.
- `current_person_confidence`는 현재 프레임의 사람 탐지 점수이며 빈자리 확률이 아닙니다.
  미탐지 유예 중에는 상태가 `OCCUPIED`여도 점수가 0일 수 있습니다.
- CPU가 매우 느려 한 번의 추론이 `--max-gap-seconds`보다 오래 걸리면 확인 불가를 기록합니다.
- 파일 입력에서는 분석 속도가 아닌 영상 시간으로 상태 전환 시간을 계산합니다.

## 예약 웹 연동용 운영 코어

`SeatMonitorCore`는 카메라 판정과 예약 상태를 섞지 않습니다. 예약만 하면
`RESERVED_WAITING`, 체크인과 사람이 모두 확인되면 `IN_USE`, 예약 없이 사람이
1분간 계속 감지되면 `UNRESERVED_OCCUPIED`가 됩니다. 예약 시작 후 15분간 체크인하지
않으면 `NO_SHOW`, 사람 없이 같은 짐이 10분간 남으면 `AWAY_WITH_BELONGINGS`, 30분이면
`CLEANUP_PENDING`입니다. 이 시간은 프레임 수가 아니라 실제 경과 시간으로 계산합니다.

운영 메타데이터는 로컬 `seat_monitor.db`에 저장됩니다. SD 카드 쓰기 횟수를 줄이기
위해 상태가 바뀔 때와 기본 60초 간격으로만 기록합니다. 웹 서버에서는 다음 메서드만
연결하면 됩니다.

`user_token`에는 학번이나 이름 대신 웹 로그인 세션에서 발급한 불투명 토큰을 넣으세요.
카메라 판정에는 사용자 토큰이 들어가지 않으며 얼굴 인식도 하지 않습니다.

- 카메라 입력: `feed_detections(...)` 또는 `feed_seat_status(...)`
- 사용자 화면: `get_user_seat_map()`, `reserve_seat(...)`, `check_in_seat(...)`
- 관리자 화면: `get_admin_dashboard()`, `request_cleanup(...)`, `resolve_cleanup(...)`

기본 `PersonDetector`는 라즈베리파이 속도를 위해 사람만 검출합니다. 짐 구분까지 시험할
때는 `detector.py`의 `SeatObjectDetector`를 사용합니다. COCO 사전 학습 클래스 중 가방,
책, 노트북, 휴대전화, 병 등을 함께 검출하며, 현장 조명과 카메라 각도에서 정확도를
반드시 확인해야 합니다. 좌석 설정 JSON에는 선택적으로 `desk_polygon`을 추가할 수
있습니다. 없으면 기존 좌석 영역을 책상 영역으로도 사용합니다.

## 튜닝

```bash
python -m seat_monitor run --preview --confidence 0.35 --imgsz 640 --fps 3 \
  --occupied-seconds 1 --empty-seconds 8 --threads 4
```

처음에는 2~4석으로 시험하세요. 멀리 있는 사람, 엎드린 자세, 칸막이 가림은 미탐지를
일으킬 수 있습니다. `--confidence`를 낮추면 미탐지가 줄지만 오탐은 늘 수 있습니다.
처리가 느리면 `--imgsz 416`으로 비교하되 작은 사람이 안 잡히는지 다시 확인합니다.
고정 카메라의 사선 하향 시점부터 시험하고, 극단적인 수직 탑뷰는 현장 검증이 필요합니다.

이 코드는 자세 추정이나 본인 확인을 하지 않습니다. 좌석 영역에 오래 서 있는 사람도
점유로 볼 수 있습니다. 사람 박스 중심으로 좌석을 나누기 어려운 설치 각도라면 구역을
다시 잡거나 향후 포즈 모델을 적용해야 합니다.

## 선택: NCNN으로 CPU 추론 최적화

먼저 기본 `.pt` 모델로 카메라와 좌석 판정을 확인한 후 Pi에서 변환합니다.

```bash
python -m pip install ncnn pnnx -c .cache/pi-constraints.txt
python -m seat_monitor prepare --ncnn --imgsz 640
python -m seat_monitor run --model models/yolo11n_ncnn_model --imgsz 640 --preview
```

변환과 실행의 `--imgsz`를 같게 유지하세요. NCNN 추가 패키지는 다운로드가 필요합니다.
내보내기 호환성 문제는 `.pt` 모델로 돌아가 먼저 시연할 수 있습니다. 모델/패키지 설치
이후 실시간 추론에는 인터넷이 필요하지 않습니다.

## PC 또는 저장 영상으로 개발하기

PC에서는 Picamera2가 필요 없습니다. Python 3.11 가상환경에 `requirements.txt`를
설치하고 모델을 준비합니다. USB 카메라 번호 또는 로컬 영상 파일을 입력합니다.

```bash
python -m seat_monitor run --source 0 --whole-frame --preview
python -m seat_monitor calibrate --source demo.mp4
python -m seat_monitor run --source demo.mp4 --preview
```

영상이 끝나면 종료하고 상태를 `UNKNOWN`으로 바꿉니다. URL/원격 영상 입력은 지원하지 않습니다.

## 데이터 처리

프로그램에는 영상 업로드, 녹화, 얼굴 인식, 개인 추적 기능이 없습니다. 모델 준비 단계만
공식 배포처에 접속하며 추론 시 Ultralytics 자동 설치·다운로드와 동기화를 비활성화합니다.
기본 실행은 프리뷰가 꺼져 있고 `--preview`를 지정하면 해당 장치의 화면에만 표시합니다.
로컬에 저장되는 것은 좌석 좌표와 최신 상태 JSON입니다. YOLO 설정 캐시는 `.cache/`에 둡니다.

## 테스트

AI 모델/카메라 설치 없이 상태 로직과 연결 처리를 검사할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

좌석 배정, 복도 제외, 겹친 좌석의 중복 배정 방지, 실제 시간 누적, 예약/체크인 분리,
알림 중복 방지, 재시작 복원, 설정 검증, 카메라 실패를 검사합니다. GitHub Actions도
같은 검사를 실행합니다.
실물 Pi/카메라의 처리 속도와 실제 열람실 정확도는 장비에서 별도로 확인해야 합니다.

실제 모델의 선택적 검사(사람이 있는 로컬 사진 필요):

```bash
python -m tests.smoke_model --image /path/to/person.jpg
```

2026-09-29 Windows/Python 3.11.9에서 자동 테스트 27개를 통과했습니다.
Ultralytics 8.3.3, PyTorch 2.4.1, OpenCV 4.10.0, NumPy 1.26.4로 공식
YOLO11n 모델을 로딩했고, Ultralytics 패키지의 `bus.jpg`에서 사람 4명,
검은색 대조 프레임에서 0명을 탐지했습니다. 이 모델 검사 동안 소켓 연결을
차단했습니다. 이는 기본 작동 확인이며, 열람실 정확도나 Pi의 성능 측정값은 아닙니다.
Picamera2 실물 입력, 좌석 설정 GUI, NCNN 변환/ARM 실행은 아직 실물 검증 전입니다.

사용한 공식 `.pt` 파일의 SHA-256:
`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`

## 파일 구성

```text
seat_monitor/
  camera.py       Picamera2 / USB / 로컬 영상 입력
  detector.py     사전 학습 모델 준비와 사람/선택적 짐 탐지
  core.py         좌석 배정, 예약 대조, 시간 정책, SQLite 저장
  calibrate.py    마우스로 좌석 영역 설정
  config.py       좌표 검증
  status.py       상태 JSON
  display.py      선택적 로컬 프리뷰
  __main__.py     명령 실행
```

오픈소스 및 모델 출처는 [THIRD_PARTY.md](THIRD_PARTY.md)에 정리했습니다.
이 프로젝트는 AGPL-3.0-only입니다. [LICENSE](LICENSE)를 참고하세요.
