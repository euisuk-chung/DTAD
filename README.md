# timeseries_anomaly
Timeseries Data Anomaly Detection

# Implementation Papers

1. **MADGAN** : [Multivariate Anomaly Detection for Time Series Data with GANs](https://arxiv.org/abs/1901.04997) [official github](https://github.com/LiDan456/MAD-GANs) [영상](https://youtu.be/Y3FMi2EW23Y?list=PLetSlH8YjIfUi6p0wdA6z2dFbXUpLor6K)
3. **TAnoGAN** : [TAnoGAN: Time Series Anomaly Detection with Generative Adversarial Networks](https://arxiv.org/pdf/2008.09567.pdf) [official github](https://github.com/mdabashar/TAnoGAN) [영상] 
(https://youtu.be/WkK52d0RWk8?list=PLetSlH8YjIfUi6p0wdA6z2dFbXUpLor6K)

# Train 

- Default 값: `main.py`의 config_args

```bash
python main.py --train --logdir $logdir --datadir $datadir --dataname $dataname --epochs $epochs --batch_size $batch_size --scale $method --gen_loss $loss --window_size $window_size 
```

# Test

- 모든 test set 시점의 Anomaly Score 계산
- 한 시점씩 새로운 데이터가 들어오는 가정하에 window 마다 맨 마지막 시점의 Anomaly Score 저장

```bash
python main.py --test --logdir $logdir --dataname $dataname --resume $version --scale $method --scale $scale --lam $lambda --optim_iter $iteration
```

# Tensorboard

```bash
tensorboard --logdir $logdir --port $port_number --host $host_ip
```
