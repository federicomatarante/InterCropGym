# Contributing Guidelines

## Code Organization Rules
### 1. Classes and Modules
- Each logical component should be a class
- One class per file
- File name should match the class name (e.g., `environment.py` contains `Environment` class)
- Use modules to group related classes (e.g., `models/` for all RL models)

### 2. Documentation
- All code MUST be documented
- Each class must have a docstring explaining:
  - What it does
  - Parameters
  - Example usage ( important! )
  - Instance variables which can be used
```python
class MyClass:
    """
    Brief description of the class\
    :param param1: Description of first parameter
    :param param2: Description of second parameter
    :ivar instanceVariable1: Description of the variable
    
    Usage:
        my_instance = MyClass(param1, param2)
        result = my_instance.my_method()
    """
```
- Each method must have a docstring with:
  - Description
  - Parameters
  - Return value
```python
def my_method(self, arg1: int) -> str:
    """
    Brief description of what the method does\
    :param arg1: Description of argument
    :return: Description of return value
    """
```

### 3. Type Hints
- All code MUST use type hints
- Use them for:
  - Method arguments
  - Return values
  - Variable annotations when type is not obvious
```python
def process_data(data: numpy.ndarray) -> Tuple[torch.Tensor, float]:
    batch: List[int] = []  # only when type is not obvious
```

### 4. Configuration Files
- All algorithms MUST use .ini files for their parameters
- Each algorithm should have its own configuration file in the `configs/` directory
- Configuration file name should match the algorithm name (e.g., `dqn.ini` for DQN algorithm)
- Structure configuration files into logical sections

Example configuration structure:
```ini
[algorithm]
learning_rate = 0.001
batch_size = 64
gamma = 0.99
; Add algorithm-specific parameters

[network]
hidden_sizes = [256, 256]
activation = relu
; Add network architecture parameters

[training]
num_episodes = 1000
max_steps = 500
save_frequency = 100
; Add training-specific parameters
```

How to implement:
```python
class DQNAgent:
    """
    Deep Q-Network agent implementation\
    :param config_path: Path to the configuration file
    
    Usage:
        # Create config file dqn.ini with parameters
        [algorithm]
        learning_rate = 0.001
        batch_size = 64
        
        # Initialize agent with config
        agent = DQNAgent("configs/dqn.ini")
        agent.train()
    """
    def __init__(self, config_path: str):
        self.config = ConfigReader(config_path)
        self.lr = float(self.config.get_param("algorithm.learning_rate"))
        self.batch_size = int(self.config.get_param("algorithm.batch_size"))
```

Configuration Guidelines:
- Keep all magic numbers in config files, not in code
- Use appropriate data types (convert strings to int/float as needed)
- Provide default values for optional parameters
- Include comments in .ini files to explain parameter purposes
- Document any dependencies between parameters

### 5. Code Style
- Follow PEP 8
- Maximum line length: 100 characters
- Use meaningful variable names
- Avoid single-letter variables except for:
  - Loop indices (i, j, k)
  - Mathematical formulas (following paper notations)

### 6. Git Workflow
- Always work on separate files
- Update the code before starting to work on a new task
- Read always the commit messages of the other person to understand updates
- Never commit directly to main
- Create a branch for each feature/fix:
  - feature/your-feature-name
  - fix/bug-description
- Commit messages should be clear and descriptive
- Push in main only when task is completed
- Make sure your code works well by testing it before pushing it

### 7. Project Structure
```
src/
├── environments/   # Environment implementations
├── agents/         # RL agents
├── utils/         # Shared utilities
├── configs/       # Configuration files
└── main.py        # Main file
```

### 8. Dependencies
- Use python 3.11
- Add all the new dependencies to requirements.txt
- Include version numbers
- Document why the dependency is needed (as a comment)