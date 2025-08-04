# spark-sched-sim

This repository is trying to imporve upon the [spark-sched-sim](https://github.com/ArchieGertsman/spark-sched-sim) repository

---

This repository is a PyTorch Geometric implementaion of the [Decima codebase](https://github.com/hongzimao/decima-sim), adhering to the Gymnasium interface. It also includes enhancements to the reinforcement learning algorithm and model design, along with a basic PyGame renderer that generates the above charts in real time.

Enhancements include:
- Continuously discounted returns, improving training speed
- Proximal Polixy Optimization (PPO), improving training speed and stability
- A restricted action space, encouraging a fairer policy to be learned
- Multiple different job sequences experienced per training iteration, reducing variance in the policy gradient (PG) estimate
- No learning curriculum, improving training speed

---

After cloning this repo, please run `pip install -r requirements.txt` to install the project's dependencies.

To start out, try running examples via `examples.py --sched [fair|decima]`. To train Decima from scratch, modify the provided config file `config/decima_tpch.yaml` as needed, then provide the config to `train.py -f CFG_FILE`.