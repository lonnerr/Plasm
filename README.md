MAE  = 1.111752e+00
RMSE = 2.752028e+00

Истинные силы:
mean |F1+F2+F3| = 4.381345e-07
max  |F1+F2+F3| = 1.575728e-05

Предсказанные силы:
mean |F1+F2+F3| = 1.285764e-01
max  |F1+F2+F3| = 9.903413e-01

ForceModel(
  (network): Sequential(
    (0): Linear(in_features=6, out_features=128, bias=True)
    (1): ReLU()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): ReLU()
    (6): Linear(in_features=128, out_features=9, bias=True)
  )
)


<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/852ae90b-3121-4b9d-8959-c95c5d2184df" />
<img width="690" height="690" alt="image" src="https://github.com/user-attachments/assets/70d21206-9f51-4b49-9e3d-d6d7356aed9a" />


