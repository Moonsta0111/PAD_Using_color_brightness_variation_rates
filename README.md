# Human Identification Using Facial Color & Brightness Variation Rates

단일 카메라 기반으로, 얼굴 섹션별 **색상·명도 변화율**을 분석하여  
영상 속 얼굴이 **실제 사람(Person)** 인지 **사진(Picture)** 인지 판별하는 안티-스푸핑(Anti-Spoofing) 코드입니다.

이 저장소는 한국통신학회 동계종합학술발표회에 게재된 논문  
**단일 카메라 기반 얼굴 섹션별 색상·명도 변화율을 이용한 사람 판별 기법**의 구현 코드입니다.

---

### ✨ 핵심 아이디어

- **추가 하드웨어(열화상 카메라, 3D 카메라 등) 없이** 일반 웹캠만 사용
- MediaPipe **FaceMesh**를 사용해 얼굴에서 **468개 도트(landmark)** 추출
- 도트를 6개 섹션으로 분할
  - 왼쪽 눈(left_eye)
  - 오른쪽 눈(right_eye)
  - 코(nose)
  - 입(mouth)
  - 왼쪽 볼(left_cheek)
  - 오른쪽 볼(right_cheek)
- 각 섹션에 대해 **연속 프레임 간 RGB 색상 차이(Euclidean distance)** 와  
  **밝기(채널 평균값) 차이**를 계산하여 변화율을 추정
- **100프레임 동안의 평균 색상·명도 변화율**을 임계값과 비교해  
  사람(Person) / 사진(Picture) 을 최종 판별

---

### 📄 논문 및 발표 정보
본 프로젝트는 2024년도 동계 KICS 포스터 세션에서 발표되었습니다. 연구 논문은 아래에서 확인할 수 있습니다.

https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12132276

### 📝 논문 인용 (Citation)

논문을 인용할 때는 다음과 같이 표기할 수 있습니다.

```text
Moonseung Choi, Sun-hong Min, Taeseok Jeong, and Yonggang Kim,
“Human Identification Method Using Color and Brightness Variation Rates
of Facial Sections Based on a Single Camera,”
Korea Information and Communications Society Winter Conference, 2025.
