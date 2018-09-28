// Experiment : Cliff-walking
// Architecture : Tabular coding of the state space with linear value functions and policy
// Learner : Actor-Critic with Eligibility traces

#include <rl.hpp>
#include <random>

#define NB_EPISODES    3000

#define paramGAMMA     .95
#define paramALPHA_V   .05
#define paramALPHA_P   .01
#define paramLAMBDA_V  .90
#define paramLAMBDA_P  .90


// The problem on which to train a controller
using Cliff     = rl::problem::cliff_walking::Cliff<20,6>;
using Param     = rl::problem::cliff_walking::Param;
using Simulator = rl::problem::cliff_walking::Simulator<Cliff,Param>;

using S = Simulator::observation_type;
using A = Simulator::action_type;

// The controller architecture
using Architecture = rl::gsl::ActorCritic::Architecture::Tabular<S, A, std::mt19937>;

// The algorithm to train the controller
using Learner      = rl::gsl::ActorCritic::Learner::EligibilityTraces<Architecture>;

int main(int argc, char* argv[]) {

    std::random_device rd;
    std::mt19937 gen(rd());

    // 1) Instantiate the simulator
    Param param;
    Simulator simulator(param);

    // 2) Instantiate the ActorCritic
    auto action_begin = rl::enumerator<A>(rl::problem::cliff_walking::Action::actionNorth);
    auto action_end = action_begin + rl::problem::cliff_walking::actionSize;
    unsigned int nb_features = Cliff::size;
    Architecture archi(nb_features,
            [](const S& s) { return s;},
            action_begin, action_end, gen);

    // 3) Instantiate the learner
    Learner learner(archi, paramGAMMA, paramALPHA_V, paramALPHA_P, paramLAMBDA_V, paramLAMBDA_P);

    // 4) run NB_EPISODES episodes
    unsigned int episode;
    unsigned int step;
    A action;
    S state, next;
    double rew;

    std::cout << "Learning " << std::endl;
    for(episode = 0 ;episode < NB_EPISODES; ++episode) {
        simulator.restart();
        learner.restart();
        step = 0;
        std::cout << '\r' << "Episode " << episode << std::flush;

        state = simulator.sense();

        while(true) {
            action = archi.sample_action(state);
            try {
                // The following may raise a Terminal state exception
                simulator.timeStep(action);

                rew = simulator.reward();
                next = simulator.sense();

                learner.learn(state, action, rew, next);

                state = next;
                ++step;
            }
            catch(rl::exception::Terminal& e) { 
                learner.learn(state, action, simulator.reward());
                break;
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Testing the learned policy" << std::endl;
    // After this training phase, we test the policy
    unsigned int nb_test_episodes = 1000;
    double cum_length = 0.0;
    for(unsigned int i = 0 ; i < nb_test_episodes; ++i) {
        simulator.restart();
        step = 0;
        state = simulator.sense();
        while(true) {
            action = archi.sample_action(state);
            try {
                // The following may raise a Terminal state exception
                simulator.timeStep(action);
                state = simulator.sense();
                ++step;
            }
            catch(rl::exception::Terminal& e) { 
                break;
            }
        }
        cum_length += step;
    }
    std::cout << "The mean length of "<< nb_test_episodes
        <<" testing episodes is " << cum_length / double(nb_test_episodes) << std::endl;

    // And let us display the action probabilities for the first state :
    std::cout << "The probabilities of the actions of the learned controller, in the start state are :" << std::endl;
    auto proba = archi.get_action_probabilities(0);
    std::cout << "P(North/s=start) = " << proba[rl::problem::cliff_walking::Action::actionNorth] << std::endl;
    std::cout << "P(East/s=start) = " << proba[rl::problem::cliff_walking::Action::actionEast] << std::endl;
    std::cout << "P(South/s=start) = " << proba[rl::problem::cliff_walking::Action::actionSouth] << std::endl;
    std::cout << "P(West/s=start) = " << proba[rl::problem::cliff_walking::Action::actionWest] << std::endl;

}
