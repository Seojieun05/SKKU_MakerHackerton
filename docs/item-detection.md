# SeatSync 사람·개인 짐 감지

기존 사람 감지에 개인 짐 감지를 추가하는 선택 실행 모듈입니다. 사람 유무와 짐 유무를 따로 기록해 착석, 짐만 남은 자리, 빈자리를 구분합니다.

## 추가 파일

- `seat_monitor/items_run.py`: 사람·짐 통합 추론과 좌석별 상태 출력
- `config/items.json`: 책상 위 짐 영역. 현장 카메라 구도에 맞게 설정해야 합니다.

기존 `seat_monitor` 패키지와 모델을 사용하며 추가 모델 다운로드는 필요하지 않습니다. 이 모듈은 기존 프로젝트에 추가해서 사용합니다.

## 감지 대상

`laptop`, `backpack`, `handbag`, `suitcase`, `cup`, `book`을 클래스 이름으로 선택합니다. 컵은 내용물이 커피인지 구분하지 않습니다. 필통과 태블릿 전용 클래스는 포함하지 않습니다. 작은 물건, 가림, 조명과 구도에 따라 미감지나 오탐이 발생할 수 있습니다.

노트북·가방 감지는 현장에서 확인했습니다. 컵·책은 코드에 추가했으며 실제 현장 감지는 아직 확인하지 않았습니다.

## 카메라와 영역

현장 입력은 960×720입니다. 미리보기와 감지의 센서 모드를 동일하게 설정해야 합니다. 현장의 IMX708에서는 `1536×864` 모드가 센서 중앙을 잘랐고, `2304×1296` 모드는 전체 범위를 사용했습니다. 추가 모듈에서는 `--sensor-width 2304 --sensor-height 1296`으로 선택할 수 있습니다. 옵션 생략 시 기존 자동 선택을 유지합니다. 미리보기 프로그램에는 동일한 센서 모드가 필요합니다.

```python
sensor={"output_size": (2304, 1296), "bit_depth": 10}
```

사람 영역은 `config/seats.json`, 물건 영역은 `config/items.json`을 사용합니다. 두 설정의 좌석 ID와 화면 비율이 일치해야 합니다. 검출 박스의 중심점이 영역 안에 들어오면 해당 좌석에 배정합니다. 사람 영역은 몸통 중심을, 짐 영역은 책상 위 물건을 포함하도록 지정합니다.

`config/items.example.json`은 이번 본선의 책상 영역 4개입니다. 카메라 위치가 같은 현장에서만 복사해서 사용하세요. 새로운 배치에서는 좌표를 다시 지정해야 합니다. 사람 영역 파일은 몸통에 맞춘 별도 설정이 필요합니다. 현재 저장소에는 현장 웹 브리지 `pi_bridge.py`가 포함돼 있지 않습니다. 브리지 설명은 현장에 이미 배포된 파일을 기준으로 합니다.

`items.json` 형식 예시입니다. 아래 영역은 예시이므로 실제 화면에 맞게 변경하세요.

```json
{
  "schema_version": 1,
  "frame_size": [960, 720],
  "seats": [
    {"id": "A01", "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]}
  ]
}
```

## 실행

다른 카메라 미리보기·감지 프로그램을 먼저 종료합니다. 한 카메라는 동시에 두 프로그램이 사용할 수 없습니다.

```bash
cd ~/SKKU_MakerHackerton
source .venv/bin/activate
python -m seat_monitor.items_run run \
  --source picamera2 \
  --width 960 --height 720 \
  --sensor-width 2304 --sensor-height 1296 \
  --imgsz 416 --fps 2 \
  --config config/seats.json \
  --items-config config/items.json
```

사람 신뢰도는 기존 기본값 0.4, 짐은 기본값 0.30입니다. `--item-confidence`로 짐 임계값을 변경할 수 있습니다. 짐 있음 확정은 2초, 짐 없음 확정은 기본 8초입니다. 영상이나 이미지는 저장하지 않습니다.

## 상태 JSON과 웹 전달

기존 `output/status.json`에 다음 좌석 필드가 추가됩니다.

| 필드 | 의미 |
| --- | --- |
| `has_person` | 기존 사람 존재 여부. `true`, `false`, `null` |
| `has_item` | 선택된 물건 존재 여부. `true`, `false`, `null` |
| `item_state` | 짐 상태: `OCCUPIED`, `EMPTY`, `UNKNOWN` |
| `current_item_confidence` | 현재 프레임의 해당 좌석 최대 짐 신뢰도 |
| `item_pending_state` | 짐 상태 전환 대기. 확정 상태와 구분 |
| `presence_state` | `PERSON_PRESENT`, `ITEMS_ONLY`, `EMPTY`, `UNKNOWN` |

기존 `state`와 `meaning: person_presence_only`는 사람 기준을 유지합니다. `state: EMPTY`여도 짐이 있을 수 있습니다. `item_meaning: selected_personal_items`와 `item_classes`가 추가됩니다. 카메라 오류나 느린 추론 때는 사람·짐을 확인 불가로 처리합니다.

기존 `pi_bridge.py`는 JSON 필드를 그대로 `POST /api/detections`에 전달합니다. 별도 터미널에서 브리지를 계속 실행해야 합니다. 인증 키는 저장소에 커밋하지 마세요.

웹에서는 `has_item` 등을 저장하고 조회 API에 포함해야 합니다. 카메라 유효시간·health 검사도 유지해야 합니다. QR 체크인과 이석·무단 점유 판단은 서버의 운영 기준과 결합합니다. 물건 감지만으로 소유자나 체크인 여부를 알 수는 없습니다.

현장 좌석 매핑은 `A01`~`A04` → 웹 `A-1`~`A-4`입니다. `A05`~`A08`은 미감시 좌석이며 빈자리로 추정하지 않습니다.

## 확인과 복구

1. 사람과 노트북 있음 → `PERSON_PRESENT`
2. 사람만 나가고 노트북 유지 → `ITEMS_ONLY`
3. 사람·짐 모두 치우고 15초 대기 → `EMPTY`
4. 컵·책을 각각 놓고 감지 여부와 좌석 배정 확인
5. 브리지 `health=ok`와 웹 표시 확인

1~3은 현장에서 확인됐습니다. 컵·책 추가 후 4~5는 재확인이 필요합니다.

기존 사람 감지로 돌아가려면 추가 실행을 종료하고 동일 카메라 인자로 `python -m seat_monitor run`을 실행합니다. 짐 관련 필드는 기존 실행에서 출력하지 않습니다.
