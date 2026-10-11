/**
 * The refresher card for each written lesson: a short recap (`tldr`) and, where one equation carries the
 * lesson, that equation in LaTeX (`eq`, rendered at build time). Shown at the top of each lesson and,
 * all together, on /refresher. Keep the symbols identical to the lesson's own "Show the math" blocks.
 */
export interface Refresher { tldr: string; eq?: string }

export const REFRESHER: Record<string, Refresher> = {
  // ---------------------------------------------------------------- math primer
  'math/vectors': {
    tldr: 'A vector is a list of numbers, drawn as an arrow or a point. You add vectors entry by entry (tip to tail), scale them by multiplying every entry, and measure their length with Pythagoras. A row of features is a vector with many entries.',
    eq: String.raw`\lVert \mathbf{a} \rVert = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}`,
  },
  'math/dot-product': {
    tldr: 'Multiply matching entries and add. The result measures how far two vectors point the same way: positive when aligned, zero when perpendicular, negative when opposed. Every neuron starts with a dot product of its weights and its inputs.',
    eq: String.raw`\mathbf{a}\cdot\mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta`,
  },
  'math/matrices': {
    tldr: 'A matrix turns one vector into another, and its columns say where each axis lands. Its output is a stack of dot products, one per row. Two matrices in a row compose into one, the order matters, and the inner sizes must match.',
    eq: String.raw`(m \times n)\,(n \times p) = (m \times p)`,
  },
  'math/derivatives': {
    tldr: 'A derivative is the slope of a function at a point: how much the output moves per small nudge of the input. With several inputs, each has a partial derivative, and together they form the gradient, which points uphill.',
    eq: String.raw`f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}`,
  },
  'math/chain-rule': {
    tldr: 'When functions are chained, their slopes multiply along the chain. When a value reaches the output along several paths, the contributions add. Multiply along a path and add across paths: that is all backpropagation does.',
    eq: String.raw`\frac{dy}{dx} = \frac{dy}{dg}\cdot\frac{dg}{dx}`,
  },
  'math/probability': {
    tldr: 'A distribution says how likely each outcome is. The expectation is where samples balance, and the variance is how spread out they are. Conditional probability changes the question, and Bayes\' rule shows why rare conditions produce many false alarms.',
    eq: String.raw`\mathbb{E}[X] = \sum_x x \, P(X = x)`,
  },

  // ---------------------------------------------------------------- Day 1
  'foundations/what-is-a-model': {
    tldr: 'A model is a function with adjustable knobs called parameters. The shape of the function is chosen by people; learning means choosing the knob values from data so the predictions come out right.',
    eq: String.raw`\hat y = w \cdot x + b`,
  },
  'foundations/data-features-labels': {
    tldr: 'A model only sees numbers. Each row is an example, each column a feature, and the label is the answer we want predicted. Categories, text, images and sound must be turned into numbers first, and a model can only be as good as its data.',
  },
  'foundations/loss-functions': {
    tldr: 'A loss turns "how wrong is the model?" into one number to minimise. Squared error punishes large mistakes heavily, so outliers pull the fit; absolute error is more forgiving. Classification uses log loss on the probability given to the right answer.',
    eq: String.raw`\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2`,
  },
  'foundations/linear-regression': {
    tldr: 'Fit a line by choosing the slope and intercept with the smallest squared error. Every line is one point on a map of the parameters, and over that map the loss is a bowl. You can solve for its bottom in closed form or walk down it with gradient descent.',
    eq: String.raw`w = \frac{\operatorname{cov}(x, y)}{\operatorname{var}(x)}, \qquad b = \bar y - w\,\bar x`,
  },
  'foundations/gradient-descent': {
    tldr: 'Repeatedly step against the gradient to go downhill on the loss. The learning rate sets the step size: too small crawls, too large overshoots or diverges. Stochastic gradient descent uses a small random batch per step, which is noisy but much cheaper.',
    eq: String.raw`\theta \leftarrow \theta - \eta\,\nabla L(\theta)`,
  },

  // ---------------------------------------------------------------- Day 2
  'foundations/logistic-regression': {
    tldr: 'Pass a linear score through the sigmoid to get a probability between 0 and 1. The weights tilt the curve and the bias slides it. Training minimises log-loss, and its gradient is simply prediction minus truth, times the input.',
    eq: String.raw`\sigma(z) = \frac{1}{1 + e^{-z}}`,
  },
  'foundations/decision-boundaries': {
    tldr: 'A classifier cuts the input space into regions, and the decision boundary is where its answer switches. A linear model can only draw a straight boundary. Adding features such as squares and products lets it bend, or even close into a loop.',
    eq: String.raw`z = b + w_1 x + w_2 y, \qquad P(\text{class 1}) = \sigma(z)`,
  },
  'foundations/softmax-cross-entropy': {
    tldr: 'Softmax exponentiates scores and normalises them into a probability distribution; temperature sharpens or flattens it. Cross-entropy is the surprise at the true class, minus the log of its probability. Its gradient with respect to the scores is p minus y.',
    eq: String.raw`p_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}, \qquad L = -\ln p_{\text{true}}`,
  },
  'foundations/overfitting-bias-variance': {
    tldr: 'Too rigid a model misses the pattern (underfitting, high bias). Too flexible a model memorises the noise (overfitting, high variance): its fit changes a lot with which points you happened to get. Test error is lowest in between.',
  },
  'foundations/regularization': {
    tldr: 'Add a penalty on the size of the weights to the loss. L2 shrinks all weights smoothly; L1 pushes some exactly to zero, giving sparse models. The strength λ trades a little bias for less variance and is chosen on validation data.',
    eq: String.raw`\text{loss}(\mathbf{w}) + \lambda \cdot \text{penalty}(\mathbf{w})`,
  },
  'foundations/train-val-test': {
    tldr: 'Training error is optimistic, so hold data out. Tune choices on a validation set and touch the test set once, at the end. Peeking at the test set or leaking information between splits makes the reported score a lie; cross-validation helps when data is scarce.',
  },

  // ---------------------------------------------------------------- Day 3
  'classic-ml/k-nearest-neighbors': {
    tldr: 'Store the training set and classify a new point by a vote of its k closest examples. Small k gives a jagged boundary, large k a smooth one. Features must be on comparable scales, and in high dimensions every point becomes far from every other.',
    eq: String.raw`d_p(a, b) = \Big(\sum_i |a_i - b_i|^p\Big)^{1/p}`,
  },
  'classic-ml/decision-trees': {
    tldr: 'A tree asks threshold questions, one feature at a time, choosing each split to remove the most impurity (information gain). Grown fully it memorises the data, so limit its depth or prune it. Its regions are axis-aligned rectangles.',
    eq: String.raw`\text{gain} = I(\text{parent}) - \left( \frac{n_L}{n} I(\text{left}) + \frac{n_R}{n} I(\text{right}) \right)`,
  },
  'classic-ml/ensembles': {
    tldr: 'Many trees beat one. A random forest averages trees grown on bootstrap samples with random feature choices, which cancels their jumpiness. Gradient boosting adds small trees in sequence, each fitting the errors the previous ones left: gradient descent in function space.',
    eq: String.raw`\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2`,
  },
  'classic-ml/svm-kernels': {
    tldr: 'A support vector machine picks the boundary with the widest margin, set by a few points called support vectors. A soft margin allows some mistakes at a price. Kernels compute dot products in a lifted space without building it, which makes the boundary curved.',
    eq: String.raw`\ell(m) = \max(0,\ 1 - m), \qquad m = y\,(\mathbf{w}\cdot\mathbf{x} + b)`,
  },
  'classic-ml/k-means': {
    tldr: 'Group unlabelled points into k clusters by alternating two steps: assign each point to its nearest centroid, then move each centroid to the mean of its points. It always converges, but to a local minimum, so the starting centroids matter (k-means++ helps).',
    eq: String.raw`J = \sum_{i=1}^{n} \lVert x_i - \mu_{a_i} \rVert^2`,
  },
  'classic-ml/pca': {
    tldr: 'Find the directions along which the data varies most: the eigenvectors of the covariance matrix, ordered by eigenvalue. Projecting onto the top few compresses the data while keeping most of its variance. Centre the data first.',
  },
  'classic-ml/classic-vs-deep': {
    tldr: 'On tabular data, gradient-boosted trees are usually the strong, cheap baseline. Neural networks pull ahead on unstructured data such as images, audio and text. Interpretability, compute and the amount of data also decide the choice.',
  },

  // ---------------------------------------------------------------- Day 4
  'neural-networks/neuron': {
    tldr: 'A neuron takes a weighted sum of its inputs plus a bias, a dot product, and passes it through an activation function. Logistic regression is exactly one neuron. The weights set the boundary\'s direction, and the bias shifts it.',
    eq: String.raw`z = w_1 x_1 + w_2 x_2 + b = w \cdot x + b`,
  },
  'neural-networks/activation-functions': {
    tldr: 'Without a non-linearity, any stack of layers collapses into a single linear layer. Sigmoid and tanh squash and saturate; ReLU keeps the positive half and has slope 0 or 1; GELU is a smooth ReLU. The slopes matter later, for backpropagation.',
    eq: String.raw`\operatorname{ReLU}(z) = \max(0, z)`,
  },
  'neural-networks/mlp': {
    tldr: 'A layer is a matrix multiply plus a bias, followed by an activation, and each layer\'s output is the next layer\'s input. Width adds units side by side; depth adds compositions. Count the parameters before making a network bigger.',
    eq: String.raw`\mathbf h^{(1)}=\tanh(W^{(1)}\mathbf x+\mathbf b^{(1)})`,
  },
  'neural-networks/hidden-layers': {
    tldr: 'A hidden layer re-represents the input: an affine map moves the space, and the activation bends it, until the classes can be separated by a straight cut. Each hidden unit is a learned feature, and deeper layers compose more bends.',
  },
  'neural-networks/universal-approximation': {
    tldr: 'Two steep sigmoids (or a few ReLUs) make a bump, and enough bumps can draw any continuous curve. So one wide hidden layer can approximate any reasonable function. The theorem says such a network exists, not how big it is or how to train it; depth often needs far fewer units.',
  },

  // ---------------------------------------------------------------- Day 5
  'neural-networks/computational-graphs': {
    tldr: 'Write a calculation as a graph of tiny operations: nodes compute, edges carry values. Each node knows its own local derivative. A value used twice has two outgoing edges, and its gradient contributions add up.',
  },
  'neural-networks/chain-rule-visually': {
    tldr: 'A gradient along a chain is the product of the local derivatives on that chain; one flat link silences the whole product. Across several paths, the products add. Sweeping backward from the output computes the gradient for every input at once.',
    eq: String.raw`\frac{dL}{dx} = f_3'(v)\; f_2'(u)\; f_1'(x)`,
  },
  'neural-networks/forward-backward-pass': {
    tldr: 'The forward pass evaluates every node in order and ends at the loss. The backward pass starts from a gradient of 1 at the loss and sends to each input the upstream gradient times the local derivative. A finite-difference check confirms the result.',
    eq: String.raw`\text{gradient sent to an input} = \text{upstream gradient} \times \text{local derivative}`,
  },
  'neural-networks/backprop-mlp': {
    tldr: 'Backpropagation assigns blame for the loss to every weight using the chain rule, one node at a time, on a real 2-3-1 network. One forward and one backward pass give all the gradients, far cheaper than nudging each weight separately. Then a gradient step lowers the loss.',
    eq: String.raw`z_j = \sum_i w_{ji}\, x_i, \qquad h_j = \tanh(z_j), \qquad \hat y = \sum_j v_j\, h_j`,
  },
  'neural-networks/autodiff': {
    tldr: 'Automatic differentiation records every operation on a tape during the forward pass, then replays it backward, each operation sending gradients to its inputs. Reverse mode gives the gradient for all weights in one backward sweep, which is why training uses it; that is what PyTorch does.',
  },
  'neural-networks/vanishing-exploding-gradients': {
    tldr: 'The gradient reaching an early layer is a product of one factor per layer. Factors below one make it vanish; factors above one make it explode. Careful initial scales, ReLU-like activations, residual connections, normalization and clipping keep it usable.',
    eq: String.raw`\frac{\partial L}{\partial \mathbf h_0} = J_1^{\top} J_2^{\top} \cdots J_L^{\top}\,\frac{\partial L}{\partial \mathbf h_L}`,
  },

  // ---------------------------------------------------------------- Day 6
  'neural-networks/optimizers': {
    tldr: 'An optimizer turns gradients into weight changes. SGD steps straight down the sampled gradient; momentum keeps a running velocity; RMSProp divides each coordinate by its typical gradient size; Adam combines both and corrects their startup bias. Which is fastest depends on the landscape.',
    eq: String.raw`\theta \leftarrow \theta - \eta\,\frac{\hat m}{\sqrt{\hat v}+\epsilon}`,
  },
  'neural-networks/learning-rate-schedules': {
    tldr: 'The learning rate scales every update; too large and training overshoots, then explodes. Schedules change it over time: warmup ramps it up at the start, and step, linear or cosine decay lower it later. Batch size changes the gradient noise, and the linear scaling rule holds only under conditions.',
    eq: String.raw`\eta_t=\eta_{min}+\frac{\eta_{max}-\eta_{min}}{2}(1+\cos(\pi p))`,
  },
  'neural-networks/weight-initialization': {
    tldr: 'Starting all weights equal makes hidden units identical forever, so start them random. The scale matters: too small and signals fade, too large and they saturate or blow up. Xavier scales by fan-in and fan-out; He uses 2 over fan-in to make up for ReLU zeroing half its inputs.',
    eq: String.raw`\operatorname{Var}(W_{ij})=\frac{2}{fan_{in}}`,
  },
  'neural-networks/normalization': {
    tldr: 'Normalization layers rescale activations to mean zero and unit variance, then apply a learned gain and shift. They differ in which numbers form a group: BatchNorm normalises each feature across the batch, LayerNorm each example across its features, and RMSNorm only rescales without centring.',
    eq: String.raw`\hat x_i = \frac{x_i - \mu}{\sqrt{v+\epsilon}}`,
  },
  'neural-networks/dropout-regularizers': {
    tldr: 'Dropout switches off random units during training and scales the survivors up, so the network cannot rely on any one of them. Weight decay shrinks weights each step, early stopping keeps the best validation checkpoint, and data augmentation adds examples whose label still fits.',
    eq: String.raw`\theta-\eta(g+\lambda\theta)=(1-\eta\lambda)\theta-\eta g`,
  },
  'neural-networks/loss-landscapes': {
    tldr: 'The loss surface has valleys, saddles and flat regions, and where you start decides which valley you reach. With many weights, saddles far outnumber bad minima. How flat a minimum looks depends on how the weights are scaled, and skip connections keep a direct path open through a deep network.',
  },

  // ---------------------------------------------------------------- Day 7
  'cnns/convolution': {
    tldr: 'Slide a small filter over an image; each output pixel is the dot product of the filter with the patch under it. Hand-made filters detect edges or blur; a CNN learns its filters. Using one set of weights everywhere (weight sharing) needs far fewer parameters than a dense layer.',
    eq: String.raw`y_{r,c} = b + \sum_{i=0}^{2}\sum_{j=0}^{2} w_{ij}\, x_{r+i,\,c+j}`,
  },
  'cnns/channels-stride-padding': {
    tldr: 'A colour image has three channels, so each filter has three slices, and a layer with many filters outputs many channels. Stride sets how far the window jumps, padding gives the border a fair share, and pooling shrinks a map without learned weights.',
    eq: String.raw`\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1`,
  },
  'cnns/receptive-fields': {
    tldr: 'A unit\'s receptive field is the patch of input that can affect it. Each 3×3 layer widens it by 2 pixels; stride and pooling make it grow faster. In practice the influence concentrates near the centre, so the effective field is smaller than the theoretical one.',
    eq: String.raw`r_l = r_{l-1} + (k_l - 1)\,j_{l-1}`,
  },
  'cnns/resnets': {
    tldr: 'From LeNet to AlexNet to VGG, networks grew deeper, until deeper plain networks started training worse, not just overfitting. ResNet adds a skip connection so each block learns only a change to its input. The identity path lets gradients reach early layers.',
    eq: String.raw`y = x + F(x)`,
  },
  'cnns/feature-visualization': {
    tldr: 'Look inside a trained CNN: first-layer filters are readable strokes and edges, and gradient ascent on the input shows what excites a channel, once jitter and other constraints stop it finding static. Saliency maps show which pixels matter for an answer, but a randomisation check shows some maps look sensible while ignoring the model.',
  },

  // ---------------------------------------------------------------- Day 8
  'sequences/why-memory': {
    tldr: 'Order carries meaning, lengths vary and context can be far away. A fixed-size feed-forward network cannot share what it learns across positions, and a plain sum of inputs cannot see order. Carrying a fixed-size state forward, step by step, solves both.',
    eq: String.raw`\mathbf h_t = f(\mathbf h_{t-1}, \mathbf x_t)`,
  },
  'sequences/rnn-cell': {
    tldr: 'An RNN cell mixes the current input with its previous hidden state and squashes the result. The same weights are reused at every step, and the hidden state becomes a running, fixed-size summary of everything read so far, in order.',
    eq: String.raw`\mathbf h_t = \tanh\!\bigl(W_{xh}\,\mathbf x_t + W_{hh}\,\mathbf h_{t-1} + \mathbf b\bigr)`,
  },
  'sequences/unrolling': {
    tldr: 'Substitute the state into itself and the loop becomes a chain with one copy of the cell per time step. It is a deep feed-forward network with tied weights and an input at every layer, and its depth is the sequence length.',
    eq: String.raw`\mathbf h_3 = f\bigl(f\bigl(f(\mathbf h_0, \mathbf x_1), \mathbf x_2\bigr), \mathbf x_3\bigr)`,
  },
  'sequences/bptt': {
    tldr: 'Backpropagation through time is ordinary backprop on the unrolled chain. Because the weights are tied, each weight\'s gradient is a sum over all time steps. Truncated BPTT cuts the chain after a fixed number of steps to save memory.',
    eq: String.raw`\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \boldsymbol\delta_t\,\mathbf h_{t-1}^{\top}`,
  },
  'sequences/vanishing-gradients-time': {
    tldr: 'Sending a gradient back through many steps multiplies it by the recurrent Jacobian again and again. When its largest singular value stays below one the gradient vanishes and long-range dependencies are lost; above one it can explode, which norm clipping caps.',
    eq: String.raw`\frac{\partial \mathbf h_T}{\partial \mathbf h_t} = \prod_{k=t+1}^{T} \operatorname{diag}\bigl(1-\mathbf h_k^2\bigr)\,W_{hh}`,
  },
  'sequences/lstm-gru': {
    tldr: 'An LSTM adds a cell state that is updated by addition, not squashing, with gates between 0 and 1 deciding what to forget, write and read. A forget gate near 1 lets the gradient pass almost unchanged across many steps. The GRU gets the same effect with two gates.',
    eq: String.raw`\mathbf c_t = \mathbf f_t \odot \mathbf c_{t-1} + \mathbf i_t \odot \mathbf g_t, \qquad \mathbf h_t = \mathbf o_t \odot \tanh(\mathbf c_t)`,
  },
  'sequences/seq2seq-attention': {
    tldr: 'An encoder reads the input sequence and a decoder writes the output. Squeezing the whole input into one fixed vector is a bottleneck. Bahdanau attention lets the decoder score every encoder state at each step and mix them, and the alignment plot shows where it looks.',
    eq: String.raw`\alpha_{ij} = \frac{\exp e_{ij}}{\sum_{k=1}^{L}\exp e_{ik}}`,
  },

  // ---------------------------------------------------------------- Day 9
  'transformers/tokenization': {
    tldr: 'Text becomes a sequence of integer IDs. Byte-pair encoding starts from bytes and repeatedly merges the most frequent pair, then encodes new text by replaying the merges in order. Vocabulary size trades sequence length against embedding size, and spaces, numbers and other scripts split in surprising ways.',
  },
  'transformers/embeddings': {
    tldr: 'Each token ID selects one row of a learned embedding table. Training moves words that are used alike close together, measured by cosine similarity, and some directions carry meaning, so offsets can be copied. Inside the model the vectors become contextual: the same word gets different vectors in different sentences.',
    eq: String.raw`\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}`,
  },
  'transformers/self-attention': {
    tldr: 'Each token makes a query, a key and a value. Query-key dot products, scaled and softmaxed, decide how much each token reads from every other, and the output is the weighted sum of values. Any token can reach any other in one step, at a cost that grows with the square of the length.',
    eq: String.raw`A = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right), \qquad \text{output}_i = \sum_j A_{ij}\,\mathbf v_j`,
  },
  'transformers/multi-head-attention': {
    tldr: 'Split the model width into several heads, each running attention in its own smaller subspace, so different heads can follow different patterns. Their outputs are concatenated and projected back with one matrix. Switching heads off one at a time shows how much the model relies on each.',
    eq: String.raw`\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O`,
  },
  'transformers/positional-encodings': {
    tldr: 'Attention alone ignores word order: shuffle the inputs and the outputs shuffle with them. Positions must be injected, as sinusoidal or learned vectors added at the input, or by RoPE, which rotates queries and keys so their dot product depends only on the distance between positions.',
    eq: String.raw`PE(pos, 2i) = \sin\!\big(pos / 10000^{2i/d}\big), \qquad PE(pos, 2i+1) = \cos\!\big(pos / 10000^{2i/d}\big)`,
  },
  'transformers/transformer-block': {
    tldr: 'A block is attention and then a position-wise MLP, each added back onto the residual stream, with normalization before (pre-norm) or after. The residual stream is a shared workspace every block reads and writes. Stacking identical blocks makes the model.',
    eq: String.raw`\mathbf x' = \mathbf x + \operatorname{MHA}\bigl(\operatorname{LN}(\mathbf x)\bigr)`,
  },
  'transformers/causal-masking': {
    tldr: 'A next-token model must not see the future, so a lower-triangular mask sets every future score to minus infinity before the softmax. Then every position can be trained in parallel on its own next token, and changing a later word cannot change an earlier output.',
    eq: String.raw`A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right), \qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}`,
  },

  // ---------------------------------------------------------------- Day 10
  'llms/pretraining': {
    tldr: 'A language model is trained to predict the next token on huge amounts of text, scored with cross-entropy (perplexity is its exponential). The data is collected, deduplicated and filtered first, and training costs about 6 operations per parameter per token. The result is a base model that continues text, not an assistant.',
    eq: String.raw`L = -\frac{1}{T-1}\sum_{t=1}^{T-1} \ln p_\theta(x_{t+1}\mid x_1,\dots,x_t), \qquad C \approx 6\,N\,D`,
  },
  'llms/sampling': {
    tldr: 'The model outputs a probability for every possible next token, and decoding picks one. Greedy decoding takes the top choice and tends to repeat. Temperature sharpens or flattens the distribution, top-k keeps the k best, and top-p keeps the smallest set whose probabilities reach p.',
    eq: String.raw`p_i(T) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}`,
  },
  'llms/scaling-laws': {
    tldr: 'Loss falls as a power law in parameters, data and compute: a straight line on log-log axes, down towards an irreducible floor. For a fixed compute budget there is a best split between model size and tokens (Chinchilla). Extrapolations can fail, and small models are often over-trained to make inference cheaper.',
    eq: String.raw`L(N, D) = E + A N^{-\alpha} + B D^{-\beta}`,
  },
  'llms/inference-kv-cache': {
    tldr: 'Generation runs the model once per new token, and most of that work repeats. Caching each layer\'s keys and values makes each new token cost one step, at the price of memory per token. Decoding is limited by memory bandwidth, so batching many requests raises throughput.',
    eq: String.raw`\mathrm{softmax}\!\left(\frac{q_t K^\top}{\sqrt{d}}\right) V`,
  },
  'llms/emergent-icl': {
    tldr: 'Models can do a task from examples placed in the prompt, without any weight update: in-context learning. During training the ability can switch on suddenly, together with a change in what the attention looks at. Induction heads are one well-studied mechanism, not the only one, and whether abilities seem to emerge suddenly with scale depends partly on the metric.',
  },
  'llms/hallucination': {
    tldr: 'Training rewards plausible text, not verified text, so a model can state falsehoods fluently, most often about rare facts. Calibration measures whether its confidence means anything. Abstaining below a threshold, supplying the fact in the prompt and checking agreement across samples all reduce it; none removes it.',
    eq: String.raw`\mathrm{ECE}=\sum_{b}\frac{|B_b|}{n}\,\bigl|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\bigr|`,
  },

  // ---------------------------------------------------------------- Day 11
  'alignment/rl-basics': {
    tldr: 'An agent acts in an environment and receives only a reward, and it learns a policy that maximises the discounted sum of future rewards. A Markov decision process formalises this. Value functions say how good states are, exploration trades off against exploitation, and the reward design shapes the behaviour.',
    eq: String.raw`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma\, G_{t+1}`,
  },
  'alignment/q-learning': {
    tldr: 'A Q-value says how good an action is in a state. Q-learning moves each estimate towards a Bellman target, the reward plus the discounted best next value, using only sampled transitions. It needs exploration (ε-greedy), and terminal states end the target.',
    eq: String.raw`Q(s,a)\leftarrow Q(s,a)+\alpha\,\bigl(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr)`,
  },
  'alignment/policy-gradients': {
    tldr: 'Adjust the policy directly: make actions that led to high return more likely. Subtracting a baseline leaves the gradient unbiased but less noisy. REINFORCE waits for whole episodes; actor-critic uses a learned critic to update after every step.',
    eq: String.raw`\hat g = \sum_t \big(G_t - b(s_t)\big)\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)`,
  },
  'alignment/deep-rl': {
    tldr: 'Replace the Q-table with a neural network and train it towards sampled Bellman targets. Naive updates chase a moving target on correlated data and can become unstable. A replay buffer mixes past experiences and a target network holds the goal still: together they make DQN.',
    eq: String.raw`y = r + \gamma\,(1-d)\,\max_{a'} Q_\theta(s',a')`,
  },
  'alignment/ppo': {
    tldr: 'PPO weighs each action by its advantage and by the ratio of new to old policy probability, clipping that ratio so one batch cannot push the policy far. It reuses each batch for a few epochs, then collects fresh data. Clipping discourages big moves but does not enforce a hard limit.',
    eq: String.raw`\min\bigl(\rho A,\ \operatorname{clip}(\rho,\ 1-\varepsilon,\ 1+\varepsilon)\,A\bigr)`,
  },

  // ---------------------------------------------------------------- Day 12
  'alignment/sft': {
    tldr: 'A pretrained model only continues text. Supervised fine-tuning keeps training it on instruction and good-response pairs, with the loss counted on the response tokens only, so it learns the behaviour of answering. It imitates its demonstrations, so it can be no better than they are.',
    eq: String.raw`L(\theta) = -\frac1N \sum_i \log \pi_\theta(y_i \mid x_i)`,
  },
  'alignment/rlhf': {
    tldr: 'People pick the better of two answers, and a reward model learns scores whose differences predict those choices. PPO then raises the reward while a KL penalty keeps the policy near the SFT model. Push too hard and the policy exploits the reward model\'s blind spots: reward hacking.',
    eq: String.raw`P(y_w \succ y_l) = \sigma\bigl(r(y_w) - r(y_l)\bigr)`,
  },
  'alignment/dpo': {
    tldr: 'The policy RLHF aims for can be written in closed form, so DPO skips the reward model and the RL loop. It trains on preference pairs with a classification-style loss that widens the gap between the chosen and rejected log-ratios against a frozen reference. That gap is an implicit reward.',
    eq: String.raw`L = -\log \sigma\bigl(\beta\,[\,(\log\pi(y_w) - \log\pi_{\text{ref}}(y_w)) - (\log\pi(y_l) - \log\pi_{\text{ref}}(y_l))\,]\bigr)`,
  },
  'alignment/constitutional-ai': {
    tldr: 'Write the standards down as a short list of principles. The model critiques and revises its own answers against them, and is fine-tuned on the revisions. Then an AI judge, not a person, labels preference pairs for the usual reward model and PPO: RLAIF. Change the principles and the behaviour changes.',
  },
  'alignment/rlvr': {
    tldr: 'When a program can check the answer, as in math or code, it can be the reward. Sample several attempts per problem and score each against the group\'s average (GRPO), so right answers are pushed up and wrong ones down. It only works where a checker exists, and a sloppy checker can be gamed.',
    eq: String.raw`\hat A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}`,
  },
  'alignment/evaluations': {
    tldr: 'A benchmark score is a sample, so it has an error bar that shrinks only like one over the square root of the number of questions. Comparing models on the same questions narrows it. Scores mislead when test questions leak into training, when people tune to the benchmark, or when it saturates.',
    eq: String.raw`\mathrm{SE} = \sqrt{\frac{p(1-p)}{n}}`,
  },

  // ---------------------------------------------------------------- Day 13
  'graphs-generative/gnn': {
    tldr: 'A graph neural network updates each node from messages sent by its neighbours, with an order-independent aggregate such as a sum or mean. Stacking k layers lets a node see k hops away. GCN averages with degree weighting, GAT learns attention over neighbours, and a readout pools nodes into one graph vector.',
    eq: String.raw`H' = \sigma\bigl(\hat D^{-1/2}\hat A\,\hat D^{-1/2}\, H\, W\bigr)`,
  },
};
