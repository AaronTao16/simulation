import os
import sys
from typing import Optional

from injector import Injector

from simu.simulation.contract.classification.perceptron_model import PerceptronModule
from simu.simulation.contract.collaborative_trainer import DefaultCollaborativeTrainerModule
from simu.simulation.contract.incentive.stakeable import StakeableImModule
from simu.simulation.data.imdb_data_loader import ImdbDataModule
from simu.simulation.logging_module import LoggingModule
from simu.simulation.simulate import Agent, Simulator

# path to run `bokeh serve` for visualization.
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

num_words = 1000
train_size: Optional[int] = None
if train_size is None:
    init_train_data_portion = 0.08
else:
    init_train_data_portion = 100 / train_size


def main():
    # agents in the simulation.
    agents = [
        # good agent
        Agent(address="Good",
              start_balance=10_000,
              mean_deposit=50,
              stdev_deposit=10,
              mean_update_wait_s=10 * 60,
              prob_mistake=0.0001,
              ),
        # bad agent
        Agent(address="Bad",
              start_balance=10_000,
              mean_deposit=100,
              stdev_deposit=3,
              mean_update_wait_s=1 * 60 * 60,
              good=False,
              )
    ]

    # set up the IMDB data, perceptron training model, and incentive mechanism related module.
    injector = Injector([
        DefaultCollaborativeTrainerModule,
        ImdbDataModule(num_words=num_words),
        LoggingModule,
        PerceptronModule,
        StakeableImModule,
    ])
    s = injector.get(Simulator)

    # accuracy on hidden test set after training with all training data:
    baseline_accuracies = {
        100: 0.6210,
        200: 0.6173,
        1000: 0.7945,
        10000: 0.84692,
        20000: 0.8484,
    }

    # start the simulation.
    s.simulate(agents,
               baseline_accuracy=baseline_accuracies[num_words],
               init_train_data_portion=init_train_data_portion,
               train_size=train_size,
               )


if __name__.startswith('bk_script_'):
    main()
