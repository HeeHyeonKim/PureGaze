import numpy as np

# 모델 클래스의 출력인 theta와 pi를 기반으로 3차원 시선 벡터를 추출한다.
def gazeto3d(gaze):
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

# 모델 클래스의 출력을 변환한 3차원 시선 벡터와 GT를 통해 시선의 오차를 추출한다.
def angular(gaze, label):
  assert gaze.size == 3, "The size of gaze must be 3"
  assert label.size == 3, "The size of label must be 3"

  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

    
