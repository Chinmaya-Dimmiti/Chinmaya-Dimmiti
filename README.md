Abstract:

When adding new capabilities to a system, the usual assumption is that training data for all tasks is always available. However, as the number of tasks grows, storing and retraining on such data becomes unavailable. Due to this, when we add new capabilities to Convolution Neural Network (CNN), the existing capabilities are unavailable. To overcome this problem, we implement Learning without Forgetting method, which uses only new task data to tarin the network while preserving the original capabilities.


Learning without Forgetting:

The purpose of Learning without Forgetting (LWF) is to learn a network that can perform well on both old tasks and new tasks when only new-task data is present. The figure above shows the working principle of LWF compared with other methods.

The key idea of LWF is inspired by knowledge Distillation.

Knowledge Distillation:

It is a method to distill the knowledge in an ensemble of cumbersome models and compress into a single model in order to make possible deployments to real-life applications.
Knowledge Distillation refers to the transfer of the learning behaviour of a model (teacher) to a student, in which, the output produced by the teacher is used as the targets for training the student. By applying this method, we can achieve results and an improvement can be obtained by distilling the knowledge in a number of models into a single model.
