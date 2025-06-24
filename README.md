# Cold-Start-Aware-Offloading-in-serverless-edge-computing
train/
├── main.py # Main program entry, starts training and evaluation processes
├── requirements.txt # Dependency specifications
├── reset_step.py # Environment and task queue initialization
├── train.py # Training logic and DQN agent implementation

##Describe what each file does

### 1. `main.py`  
- **Reinforcement Learning Algorithm**: Multi-agent Deep Q-Network (DQN)  
- **Scenario**: IoT task offloading with edge-cloud collaboration  
- **Features**:  
  - Task and system status observations  
  - Cold start delay modeling  
  - Edge and cloud capacity constraints  
- **Optimization Goals**:  
  - Minimize delay, energy consumption, and cost  
  - Improve resource utilization  
  - Reduce cold start frequency  

---

### 2. `requirements.txt`  
- TensorFlow 2.10 (using `tensorflow.compat.v1` for TensorFlow 1.x compatibility)  
- Includes dependencies such as Pandas, NumPy, gRPC, OAuth, etc.  
- The Graphviz tool must be pre-installed and added to system environment variables  

---

### 3. `reset_step.py`  
- Initializes task queues including:  
  - IoT computation queues  
  - Transmission queues  
  - Edge/Cloud execution queues  
- Initializes various statistical indicators such as:  
  - Pending task counts  
  - Transmission data sizes  
  - Cold start records, etc.  
- Constructs the initial observation vector (`observation_all`) for each IoT device, used as input to the DQN  
- Outputs the initial state matrix of the environment  

---

### 4. `train.py`  
- **`reward()` function**: Defines immediate reward based on delay, energy consumption, and cost penalties  
- **`DeepQNetwork` class**: Implements the DQN agent using TensorFlow 1.x API (via `tensorflow.compat.v1`)  
- **`train()` function**: Main training loop that iterates over all IoT devices, performs action selection, and trains the DQN  
- Neural network architecture consists of two hidden layers (`l1` and `lm1`), outputting Q-values  
- The state vector (`observation_all`) has a large dimension, with feature count equal to `n_features`  
- Experience replay buffer shape: `[memory_size, 2 * n_features + 2]`, storing:  
  - State, Action, Reward, Next state  

---

## How to Run

### Step 1: Prepare the Environment  
- Make sure the task trace JSON files are placed in the designated directory  
- Ensure all dependencies are installed, including Graphviz configured in system environment variables  

### Step 2: Train the Model  
```bash
python train/train.py
Initialize system architecture parameters (number of IoT devices, edge servers, cloud servers, etc.)

Load task data in JSON format

Create an independent DQN agent for each IoT device

Start the training process and output the results as a record array


