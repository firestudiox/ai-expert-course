import React from "react";
import "./styles.css";

const V1 = () => {
  return (
    <React.Fragment>
      <h1>RL Theory</h1>
      <div className="wrapper">
        {/* title */}
        <div className="box box0 ">
          <div className="nested">
            <h3>Intro</h3>
          </div>
          <div className="nested">
            <h3> Markov Decision Process</h3>
            <h5>MDP</h5>
            <h5>Value Function</h5>
            <ul>
              <li>State-value function</li>
              <li>Action-value function</li>
            </ul>
          </div>
          <div className="nested">
            <h3>Bellman Equation </h3>
            <h5> Bellman Expectation Equation</h5>
            <h5> Bellman Optimality Equation</h5>
          </div>
          <div className="nested">
            <h3>Dyanmic Programming</h3>
            <h5> Policy Iteration</h5>
            <h5>Value Iteration</h5>
          </div>
          <div className="nested">
            <h3>Monte-Carlo Learning</h3>
            <h5>Monte-Carlo prediciton</h5>
            <h5>Monte-Carlo Control</h5>
          </div>
          <div className="nested">
            <h3>Temporal Difference Learning</h3>
            <h5>TD Prediction</h5>
            <h5>TD Control</h5>
            <h5> Eligibility Traces</h5>
          </div>
          <div className="nested">
            <h3>Q-Learning: Off-Policy Control</h3>
            <h5>Importance Sampling</h5>
            <h5>Q Learning</h5>
          </div>
          <div className="nested">
            <h3>Value Function Approximation</h3>
            <h5>Value Function Approximation</h5>
            <h5>Stochastic Gradient Descent</h5>
            <h5>Learning with Function Approximator</h5>
          </div>
          <div className="nested">
            <h3>Deep Q-Network</h3>
            <h5>Neural Network</h5>
            <h5>Deep Q Networks</h5>
          </div>
          <div className="nested">
            <h3>Policy Gradient</h3>
            <h5>1. Numerical Methods</h5>
            <h5>2. Monte-Carlo Policy Gradient</h5>
            <h5>3. Actor-Critic Policy Gradient</h5>
          </div>
          <div className="nested">
            <h3>Newest Concepts???</h3>
            <h5>...</h5>
          </div>
        </div>
        {/* Theory: Main Concept */}
        <div className="box box1 ">
          <div className="nested">00.Intro</div>
          <div className="nested">
            <h5>MDP</h5>
            <h5>Value Function</h5>
            <ul>
              <li>State-value function</li>
              <li>Action-value function</li>
            </ul>
          </div>
          <div className="nested">
            <div>Agent는 value function을 가지고 자신의 행동을 선택</div>
            <h5> Bellman Expectation Equation</h5>
            <p>
              현재 state의 value function과 next state의 value function 사이의
              관계식
            </p>
            <p>expectation의 형태로 표현된 Bellman equation: </p>
            <p>
              Backup:미래의 값들(next state-value function)으로 현재의 value
              function을 구한다는 것{" "}
            </p>
            <ul>
              <li>Full-width backup: dynamic programming</li>
              <li>sample backup: reinforcement learning</li>
            </ul>
            <h5>Optimal value function</h5>
            <p>
              강화학습의 목적: accumulative future reward를 최대로 하는 policy를
              찾는 것
            </p>
            <p>
              optimal state-value function: 현재 state에서 policy에 따라서
              앞으로 받을 reward들이 달라지는데 그 중에서 앞으로 가장 많은
              reward를 받을 policy를 따랐을 때의 value function
            </p>
            <p>
              optimal action-value function의 값을 안다면 단순히 q값이 높은
              action을 선택해주면 되므로 이 최적화 문제는 해결
            </p>
            <p>
              optimal policy는 (s,a)에서 action-value function이 가장 높은
              action만을 고르기 때문에 deterministic
            </p>
            <h5> Bellman Optimality Equation</h5>
            <p>
              Bellman optimality equation는 위의 optimal value function 사이의
              관계를 나타내주는 식
            </p>
            <p>
              Bellman equation을 통해서 iterative하게 MDP의 문제를 푸는 것이
              Dynamic Programming
            </p>
            <p>
              Bellman equation은 dynamic programming같이 discrete한 time에서의
              최적화 문제에 적용되는 식
            </p>
          </div>
          {/* DP */}
          <div className="nested">
            <h5>Dynamic Programming</h5>

            <div>planning vs. learning</div>
            <ul>
              <li>
                Planning이란 environment의 model을 알고서 문제를 해결하는 것
              </li>
              <li>
                Learning이란 environment의 model을 모르지만 상호작용을 통해서
                문제를 해결하는 것
              </li>
            </ul>
            <div>Reinforce learning</div>
            <ul>
              <li>환경을 알 수 없다.</li>
              <li>Agent는 환경과 상호작용한다</li>
              <li>Agent는 정책을 개선한다improve</li>
            </ul>
            <div>Planning</div>
            <ul>
              <li>환경의 model을 이미 알고 있다.</li>
              <li>Agent는 model을 가지고 계산을 수행한다.</li>
              <li>Agent는 정책을 개선한다improve</li>
            </ul>

            <div>
              DP: Planning으로서 Environment의 model(reward, state transition
              matrix)에 대해서 안다는 전제로 (Bellman equation을 사용해서)
              문제를 푸는 방법
            </div>
            <h5>Prediction & Control</h5>
            <div>
              Dynamic Programming은 (1) Prediction (2)Contro의 두 step을
              반복하는 것으로 이루어지며 optimal policy를 구하는 것.
            </div>
            <ul>
              <li>
                prediction: optimal하지 않는 어떤 policy에 대해서 value
                function을 구하고 (->Policy evaluation)
              </li>
              <li>
                control: 현재의 value function을 토대로 더 나은 policy를 구하고
                (->improve)
              </li>
            </ul>

            <h5>Policy evaluation</h5>
            <div>
              Policy evaluation은 prediction 문제를 푸는 것으로서 현재 주어진
              policy에 대한 true value functiond을 구하는 것이고 Bellman
              equation을 사용
            </div>
            <div>
              현재 상태의 value function을 update하는데 reward와 next state들의
              value function을 사용하는 것
            </div>
            <div> DP = evaluation -> improve</div>
            <div>
              evaluation은 현재의 policy가 얼마나 좋은가를 판단하는 것이고 판단
              기준은 그 policy를 따라가게 될 경우 받게 될 value function
            </div>
            <h5>Policy Iteration</h5>
            <div>
              Policy improvement: 해당 policy에 대한 True value를 얻었으면,
              policy를 더 나은 policy로 update하여 optimal policy에 가까워
              지도록 하는 과정
            </div>
            <div>
              방법: greed improvement:다음 state중에서 가장 높은 value
              function을 가진 state로 가는 것{" "}
            </div>
            <h3>Value Iteration:</h3>
            <div>
              optimal value function들 사이의 관계식인 Bellman Optimality
              Equation을 사용
            </div>
            <div>
              evaluation을 단 한 번만 하는 것이 value iteration: 현재 value
              function을 계산하고 update할 때 max를 취함으로서 greedy하게
              improve하는 효과
            </div>
            <div>한 번의 evaluation + improvement = value iteration</div>

            <h5>Sample Backup</h5>
            <div>
              DP는 MDP에 대한 정보를 다 가지고 있어야 optimal policy를 구할 수
              있다. 또한, DP는 full-width backup(한 번 update할 때 가능한 모든
              successor state의 value function을 통해 update하는 방법)을
              사용하고 있기 때문에 단 한 번의 backup을 하는 데도 많은 계산을
              요구. So? 실사용에 어려움!!!
            </div>
            <div>어떻게 해결? sample back-up!</div>
            <div>
              모든 가능한 successor state와 action을 고려하는 것이 아니고
              Sampling을 통해서 한 길만 가보고 그 정보를 토대로 value function을
              업데이트 --{`>`} model-free
            </div>
            <div>
              DP의 방법대로 optimal한 해를 찾으려면 매 iteration마다 Reward
              function과 state transition matrix를 알아야 하는데 sample backup의
              경우에는 아래 그림과 같이 {`<`}S,A,R,S'{`<`}을 training set으로
              실재 나온 reward와 sample transition으로서 그 두 개를 대체
            </div>
            <div>
              MDP라는 model을 몰라도 optimal policy를 구할 수 있게 되고
              "Learning"이 되는 것
            </div>
            <div>
              DP를 sampling을 통해서 풀면서부터 "Reinforcement Learning"이 시작
            </div>
          </div>
          {/* MC */}
          <div className="nested">
            <h3>Monte-Carlo Prediction</h3>
            <h5>Model Free</h5>
            <div>
              Environment의 model을 모르고 Trial and error를 통해서 실재로
              경험한 정보들로서 update를 하는 sample backup
            </div>
            <ul>
              <li>
                model-free prediction: 현재의 policy를 바탕으로 움직여보면서
                sampling을 통해 value function을 update
              </li>
              <li>
                model-free control: model-free prediction + policy를 update
              </li>
            </ul>
            <h3> Sampling을 통해서 학습하는 model-free 방법</h3>
            <ol>
              <li>Monte-Carlo</li>
              <li>Temporal Difference</li>
            </ol>
            <h5>Monte-Carlo</h5>
            <div>
              Monte-Carlo: 강화학습에서 "averaging complete returns"하는 방법
            </div>
            <div>
              TD와 MC가 나누어지는 것은 expected accumulative future reward로서
              지금 이 state에서 시작해서 미래까지 받을 기대되는 reward의 총합인
              value function을 estimation하는 방법
            </div>
            <div>
              어떻게 구하는가? episode를 끝까지 가본 후에 받은 reward들로 각
              state의 value function들을 거꾸로 계산. 결과적으로 MC는 끝나지
              않는 episode에서는 사용할 수 없는 방법
            </div>
            <h5>First-Visit MC vs Every-Visit MC</h5>
            <div>한 episode내에서 어떠한 state를 두 번 방문한다면?</div>
            <ul>
              <li>First-visit Monte-Carlo Policy evaluation</li>
              <li>Every-visit Monte-Carlo Policy evaluation</li>
            </ul>
            <h5>Incremental Mean</h5>
            <div>
              여러개를 모아놓고 한 번에 평균을 취하는 것이 아니고 하나 하나
              더해가며 평균을 계산.
            </div>
            <h5>ackup Diagram</h5>
            <div>
              DP에서는 one-step backup에서 그 다음으로 가능한 모든 state들로
              가지가 뻗었었는데 MC에서는 sampling을 하기 때문에 하나의 가지로
              terminal state까지 graphing
            </div>
            <div>
              Monte-Carlo는 random process를 포함한 방법. episode마다 update하기
              때문에 경우에 따라서 전혀 다른 experience. 따라서 variance가
              높으나 편향은 적어서 bias는 낮은 편.
            </div>
            <h3>Monte-Carlo Control</h3>
            <h5> Monte-Carlo Policy Iteration</h5>
            <ul>
              <li>Monte-Carlo Policy Evaluation = Prediction</li>
              <li>DP: evaluation + Improvement = Policy Iteration</li>
              <li>
                MC: Monte-Carlo Policy Evaluation + Policy Improvement =
                Monte-Carlo Policy Iteration
              </li>
            </ul>
            <h5>Monte-Carlo Control</h5>
            <div>Monte-Carlo Policy Iteration의 문제점</div>
            <ul>
              <li>Value function</li>
              <li>Exploration</li>
              <li>Policy Iteration</li>
            </ul>
            <h5>(1) Value function</h5>
            <div>
              value function 대신에 action value function을 사용하면 문제없이
              model-free
            </div>
            <h5>(2) Exploration</h5>
            <div>local obtimum에 봉착: ε-greedy policy improvement:</div>
            <h5>(3) Policy Iteration</h5>
            <div>
              DP의 Vaule Iteration과 마찬가지로 evaluation과정을 줄임으로서
              Monte-Carlo policy iteration에서 Monte-Carlo Control이 된다.
            </div>
            <h5>GLIE: Greedy in the Limit with Infinite Exploration</h5>
            <div>
              학습을 해나감에 따라 충분한 탐험을 했다면 greedy policy에 수렴하는
              것
            </div>
          </div>
          {/* TD */}
          <div className="nested">
            <h2>TD Prediction</h2>
            <h5>Temporal Difference</h5>
            <div>
              MC:online으로 바로바로 학습할 수가 없고 꼭 끝나는 episode여야 한다
            </div>
            <div>
              <p>
                episode가 끝나지 않더라고 DP처럼 time step마다 학습할 수는
                없을까? TD!
              </p>
              <p>TD: MC와 DP를 섞은 것</p>
              <ul>
                <li>TD = Monte-Carlo + DP</li>
                <li>MC처럼 raw experience로부터 학습</li>
                <li>DP처럼 time step마다 학습</li>
                <li>
                  bootstrap: 주변의 state들의 value function을 사용: Bellman
                  Equantion
                </li>
              </ul>
            </div>
            <h5> TD(0)</h5>
            <h5>MC vs TD</h5>
            <ul>
              <li>
                MC: 목적지에 도착한다음에 각각의 state에서 예측했던 value
                function과 실재로 받은 return을 비교해서 update
              </li>
              <li>
                TD: 한 스텝 진행을 하면 아직 도착을 하지 않아서 얼마가 걸릴지는
                정확히 모르지만 한 스텝 동안 지났던 시간을 토대로 value
                function을 update
              </li>
            </ul>
            <div>
              실재로 도착을 하지 않아도, final outcome을 모르더라고 학습할 수
              있는 것이 TD의 장점이며 매 step마다 학습할 수 있다는 것도 장점
            </div>
            <h5>Bias/Variance Trade-Off</h5>
            <h5>
              TD는 한 episode안에서 계속 업데이트를 하는데 보통은 그 전의 상태가
              그 후에 상태에 영향을 많이 주기 때문에 학습이 한 쪽으로
              편향(bias가 높다!)
            </h5>
            <h3>TD Control</h3>
            <h5>Sarsa</h5>
            <div>
              model-free control이 되기 위해서는 action-value function을 사용
            </div>
            <div>
              TD(0)의 식에서 value function을 action value function으로
              바꾸어주면 Sarsa
            </div>
            <div>
              현재 state-action pair에서 다음 state와 다음 action까지를 보고
              update
            </div>
            <div>
              on-policy TD control algorithm으로서 매 time-step마다 현재의 Q
              value를 imediate reward와 다음 action의 Q value를 가지고 update
            </div>
            <div>
              policy는 따로 정의되지는 않고 이 Q value를 보고 ε--greedy하게
              움직이는 것 자체가 policy
            </div>
            <h3>Eligibility Traces</h3>
            <h5>n-step TD</h5>
            <h5> Forward-View of TD</h5>
            <h5>Backward-View of TD</h5>
            <h5>Sarsa</h5>
          </div>
          <div className="nested">
            <h3>Q learning: Off-Policy Control</h3>
            <div>
              Monte-Carlo Control과 Temporal-Difference Control==> on-policy
              reinforcement learning:
            </div>
            <h3>Importance Sampling</h3>
            <div>
              on-policy 한계: 탐험 문제-현재알고있는 정보에 대해 greedy로
              policy를 정해버리면 optimal에 가지 못 할 확률이 커지기 때문에
              에이전트는 항상 탐험이 필요
            </div>
            <div>on-policy vs. off-policy</div>
            <ul>
              <li>on-policy: 움직이는 policy와 학습하는 policy가 같은 것</li>
              <li>off-policy: 두개의 policy를 분리시킨 것</li>
            </ul>
            <div>off-policy의 장점</div>
            <ul>
              <li>다른 agent나 사람을 관찰하고 그로부터 학습</li>
              <li>이전의 policy들을 재활용하여 학습</li>
              <li>탐험을 계속 하면서도 optimal한 policy를 학습(Q-learning)</li>
              <li>하나의 policy를 따르면서 여러개의 policy를 학습</li>
            </ul>
            <h5>Importance sampling</h5>
            <div>
              "importance sampling"이라는 개념은 원래 통계학에서 사용하던
              개념으로 아래와 특정한 분포의 값들을 추정하는 기법
            </div>
            <div>
              Off-policy learning을 할 때 Importance sampling말고 다른 방법을
              생각할 필요가 있습니다. 바로 여기서 유명한 Q learning알고리즘이
              탄생
            </div>

            <h3>Q learning</h3>
            <div>
              Off-Policy learning을 하는데 가장 좋은 알고리즘은 Q Learning
            </div>
            <div>방범</div>
            <ul>
              <li>
                현재 state S에서 action을 선택하는 것은 behaviour policy를
                따라서 선택
              </li>
              <li>
                TD에서 udpate할 때는 one-step을 bootstrap하는데 이 때 다음
                state의 action을 선택하는 데는 behaviour policy와는 다른
                policy(alternative policy)를 사용하면 Importance Sampling이
                필요없다
              </li>
            </ul>
            <h3>Off-Policy Control with Q-Learning</h3>
            <h3>Sarsa vs Q-learning</h3>
          </div>

          <div className="nested">
            <h3>Value Function Approximation</h3>
            <h5>Tabular Methods</h5>
            <h5>Parameterizing value function</h5>
            <div>
              input: state ==> w라는 parameter로 조정되는 함수 ==> output:
              action value function
            </div>
            <div>
              학습을 통해서 Q function을 update하는 것이 아니고 w라는
              parameter를 업데이트
            </div>
            <h3> Stochastic Gradient Descent</h3>
            <h5>Gradient Descent</h5>
            <div>
              어떻게 parameter를 update할 수 있을까? Stochastic Gradient
              Descent방법을 활용하여 value function의 parameter를 update
            </div>
            <div>
              w로 표현하는 함수는 어떠한 update의 목표로서 보통은 내가 원하는
              target과 자신의 error로 설정해서 그 error를 최소화하는 것을 목표
            </div>
            <div>
              update를 하려면 어느방향으로 가야 그 error가 줄어드는 지
              알아야하는 데 그것을 함수의 미분(gradient)을 취해서 알 수 있다.
            </div>
            <div>
              gradient자체는 경사이기 때문에 곡면에서 보자면 위로 올라가는
              방향이므로 -를 곱해서 그 반대 방향으로 내려감으로서(descent)
              조금씩 error를 줄여나가는 것
            </div>
            <h3> Learning with Function Approximator</h3>
            <h5>Action-value function appoximation</h5>
            <h5>Batch Methods</h5>
            <div>
              training data(agent가 경험한 것)들을 모아서 한꺼번에 update하는
              것이 "Batch Methods"
            </div>

            <h5>Experience Replay</h5>
          </div>

          <div className="nested">
            <h3>Deep Q-Network?</h3>
            <div>
              action-value function(q-value)를 approximate하는 방법으로 deep
              neural network를 택한 reinforcement learning방법이 Deep
              Reinforcement Learning(deepRL)
            </div>
            <h5>Artificial Neural Networks (ANN)</h5>
            <h5>SGD(Stochastic Gradient Descent) and Back-Propagation</h5>
            <div>
              action value function뿐만 아니라 policy 자체를 approximate할 수도
              있는데 그 approximator로 DNN을 사용해도 DeepRL
            </div>
            <h3>Neural Network</h3>
            <h5>Activation Function</h5>
            <div>
              y = Wx + b: input signal들과 weight가 곱해지고 bias가 더해진 net
              input signal이 node를 activate시키는데 그 형식을 function으로 정의
            </div>
            <div>
              activation function의 가장 간단한 형태는 들어온 input들의 합이
              어떤 Threshold보다 높으면 1이 나오고 낮으면 0이 나오는 형태. 이런
              형태의 activation function의 경우에는 미분이 불가능하고 따라서
              gradient descent를 못 쓰기 때문에 그 이외의 미분가능 함수를 사용
            </div>
            <ul>
              <li>sigmoid function</li>
              <li>tanh function</li>
              <li>absolute function</li>
              <li>ReLU function</li>
            </ul>
            <h5>SGD and Back-Propagation</h5>
            <div>
              강화학습의 목표는 optimal policy를 구하는 것이고 각 state에서
              optimal한 action value function을 알고 있으면 q값이 큰 action을
              취하면 되는 것이므로 결국은 q-value를 구하면 강화학습 문제는 해결{" "}
            </div>
            <div>weight bias를 어떻게 update? SGD</div>
            <h5> Back-Propagation</h5>
            <div>
              parameter를 SGD로 update할 때는 그 반대 방향으로 가며 갱신
            </div>
            <h3>Deep Q Networks</h3>
            <div>atari...</div>
          </div>
          {/* Policy Gradient */}
          <div className="nested">
            <h3>Policy Gradient</h3>
            <h5>Value-based RL VS Policy-based RL</h5>
            <div>
              Policy-based RL은 Policy자체를 approximate해서 function
              approximator에서 출력되는 것이 value function이 아니고
              policy자체가 출력된다. Policy자체를 parameterize하는 것
            </div>
            <div>Value-based RL 방식에는 두 가지 문제</div>
            <ul>
              <li>Unstable</li>
              <li>Stochastic Policy</li>
            </ul>
            <h5>Policy Objective Function</h5>
            <div>
              학습은 policy를 approximate한 parameter들을 update해나가는 것
            </div>
            <div>
              이 parameter을 update하려면 기준이 필요한 데 DQN에서는 TD error를
              사용
            </div>
            <dt>Policy Gradient에서는 Objective Function을 정의</dt>
            <dd>state value, </dd>
            <dd>average value, </dd>
            <dd>average reward per time-step</dd>
            <h3>Monte-Carlo Policy Gradient: REINFORCE </h3>
            <h3>Actor-Critic Policy Gradient</h3>
            <div>Policy Gradient</div>
          </div>
          <div className="nested">
            <h3>newest ideas</h3>
          </div>
        </div>
        <div className="box box2 ">
          <div className="nested">Intro</div>
          <div className="nested">MDP</div>
          <div className="nested">Bellman Eq.</div>
          <div className="nested">
            <h5>DP의 한계</h5>
            <ul>
              <li>(1) Full-width Backup --> expensive computation</li>
              <li>(2) Full knowledge about Environment</li>
              <li>
                sample backup: Trial and error로 실재로 경험한 정보들로서 update
              </li>
            </ul>
          </div>
          <div className="nested">
            <h5>MC의 한계</h5>
            <ul>
              <li>Value function</li>
              <li>Exploration</li>
              <li>Policy Iteration</li>
              <li>*** online-->TD</li>
            </ul>
          </div>
          <div className="nested">Temporal Difference Learning</div>
          <div className="nested">Off-Policy Control</div>
          <div className="nested">Value Function Approximation</div>
          <div className="nested">Deep Q-Network</div>
          <div className="nested">Policy Gradient</div>
          <div className="nested">Summary and Future???</div>
        </div>
        {/* 해결방안 */}
        <div className="box box3 ">
          <div className="nested">
            <div>training_data = ...</div>
            <div>validation_data = ...</div>
            <div>..</div>
          </div>
          <div className="nested">
            <div>model=Sequntial()</div>
            <div>model.add(Flatten(input_shape=(1,)) --- a</div>
            <div>model.add(Dense(2, activation = 'sigmoid') --- b</div>
            <div>model.add(Dense(1, activation = 'relu'))</div>
            <div>model.add(Dense(1, activation = 'softmax'))</div>
            ...
            <div>
              a+b: model.add(Dense(2, activation='sigmoid', input_shape(1,))
            </div>
          </div>
          <div className="nested">dd</div>
          <div className="nested">
            <h5>문제해결</h5>
            <ul>
              <li>(1) Monte-Carlo</li>
              <li>(2) Temporal Difference</li>
            </ul>{" "}
          </div>
          <div className="nested">
            <h5>해결방안</h5>
            <ul>
              <li>value function -> action value function을 사용</li>
              <li> policy improve--> ε-greedy policy improvement</li>
              <li>Policy Iteration -->Monte-Carlo Control</li>
            </ul>
          </div>
        </div>
        {/* ... */}
        <div className="box box4 ">
          <div className="nested">Note 1</div>
          <div className="nested">Note 2</div>
          <div className="nested">Note 3</div>
          <div className="nested">Note 4</div>
          <div className="nested">Note 5</div>
        </div>
        <div className="box box5 ">
          <div className="nested">Summary 1</div>
          <div className="nested">Summary 2</div>
          <div className="nested">Summary 3</div>
          <div className="nested">Summary 4</div>
          <div className="nested">Summary 5</div>
        </div>
      </div>
    </React.Fragment>
  );
};

export default V1;
