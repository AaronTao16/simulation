import json
import logging
import os
import random
import time
import numpy as np
from dataclasses import asdict, dataclass
from functools import partial
from itertools import cycle
from logging import Logger
from platform import uname
from queue import PriorityQueue
from threading import Thread
from typing import List
from bokeh import colors
from bokeh.document import Document
from bokeh.io import export_png
from bokeh.models import AdaptiveTicker, ColumnDataSource, FuncTickFormatter, PrintfTickFormatter
from bokeh.plotting import curdoc, figure
from injector import inject
from tornado import gen
from tqdm import tqdm

from simu.simulation.contract.balances import Balances
from simu.simulation.contract.collaborative_trainer import CollaborativeTrainer
from simu.simulation.contract.incentive.prediction_market import MarketPhase, PredictionMarket
from simu.simulation.contract.objects import Address, Msg, RejectException, TimeMock
from simu.simulation.data.data_loader import DataLoader
from simu.simulation.data.featuremapping.feature_index_mapper import FeatureIndexMapper

# Agent class
@dataclass
class Agent:
    address: Address
    start_balance: float
    mean_deposit: float
    stdev_deposit: float
    mean_update_wait_s: float
    stdev_update_wait_time: float = 1
    pay_to_call: float = 0
    good: bool = True
    prob_mistake: float = 0
    calls_model: bool = False

    def __post_init__(self):
        assert self.start_balance > self.mean_deposit

    def __lt__(self, other):
        return self.address < other.address

    def get_next_deposit(self) -> int:
        while True:
            result = int(random.normalvariate(
                self.mean_deposit, self.stdev_deposit))
            if result > 0:
                return result

    def get_next_wait_s(self) -> int:
        while True:
            result = int(random.normalvariate(
                self.mean_update_wait_s, self.stdev_update_wait_time))
            if result >= 1:
                return result

# simulation process
class Simulator(object):

    @inject
    def __init__(self,
                 balances: Balances,
                 data_loader: DataLoader,
                 simu: CollaborativeTrainer,
                 feature_index_mapper: FeatureIndexMapper,
                 logger: Logger,
                 time_method: TimeMock,
                 ):

        self._balances = balances
        self._data_loader = data_loader
        self._simu = simu
        self._feature_index_mapper = feature_index_mapper
        self._logger = logger
        self._time = time_method
        self._warned_about_saving_plot = False

    # save result plot
    def save_plot_image(self, plot, plot_save_path):
        try:
            export_png(plot, filename=plot_save_path)
        except Exception as e:
            if self._warned_about_saving_plot:
                return
            show_error_details = True
            message = "Could not save picture of the plot."
            try:
                # Check if in WSL.
                show_error_details = not (
                    'microsoft' in uname().release.lower())
            except:
                pass
            if show_error_details:
                self._logger.exception(message, exc_info=e)
            else:
                self._logger.warning(f"{message} %s", e)
            self._warned_about_saving_plot = True

    """
        :param agents: The agents list
        :param baseline_accuracy: baseline accuracy of the model
        :param init_train_data_portion: pre-train data
        :param pm_test_sets: prediction market incentive mechanism.
        :param accuracy_plot_wait_s: time to plot accuracy
        :param train_size: training data size
        :param test_size: test data size
        :param filename_indicator
    """
    def simulate(self,
                 agents: List[Agent],
                 baseline_accuracy: float = None,
                 init_train_data_portion: float = 0.1,
                 pm_test_sets: list = None,
                 accuracy_plot_wait_s=2E5,
                 train_size: int = None, test_size: int = None,
                 filename_indicator: str = None
                 ):

        assert 0 <= init_train_data_portion <= 1

        save_data = dict(agents=[asdict(a) for a in agents],
                         baselineAccuracy=baseline_accuracy,
                         initTrainDataPortion=init_train_data_portion,
                         accuracies=[],
                         balances=[],
                         )
        time_for_filenames = int(time.time())
        save_path = f'saved_runs/{time_for_filenames}-{filename_indicator}-simulation_data.json'
        model_save_path = f'saved_runs/{time_for_filenames}-{filename_indicator}-model.json'
        plot_save_path = f'saved_runs/{time_for_filenames}-{filename_indicator}.png'
        self._logger.info("Saving run info to \"%s\".", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Set up plots.
        doc: Document = curdoc()
        doc.title = "Simulation"

        plot = figure(title="Balances & Accuracy on Hidden Test Set")
        plot.width = 800
        plot.height = 600

        plot.xaxis.axis_label = "Time (days)"
        plot.yaxis.axis_label = "Percent"
        plot.title.text_font_size = '20pt'
        plot.xaxis.major_label_text_font_size = '20pt'
        plot.xaxis.axis_label_text_font_size = '20pt'
        plot.yaxis.major_label_text_font_size = '20pt'
        plot.yaxis.axis_label_text_font_size = '20pt'

        plot.xaxis[0].ticker = AdaptiveTicker(base=5 * 24 * 60 * 60)
        plot.xgrid[0].ticker = AdaptiveTicker(base=24 * 60 * 60)

        balance_plot_sources_per_agent = dict()
        good_colors = cycle([
            colors.named.green,
            colors.named.lawngreen,
            colors.named.darkgreen,
            colors.named.limegreen,
        ])
        bad_colors = cycle([
            colors.named.red,
            colors.named.darkred,
        ])
        
        for agent in agents:
            source = ColumnDataSource(dict(t=[], b=[]))
            assert agent.address not in balance_plot_sources_per_agent
            balance_plot_sources_per_agent[agent.address] = source
            if agent.calls_model:
                color = 'blue'
                line_dash = 'dashdot'
            elif agent.good:
                color = next(good_colors)
                line_dash = 'dotted'
            else:
                color = next(bad_colors)
                line_dash = 'dashed'
            plot.line(x='t', y='b',
                      line_dash=line_dash,
                      line_width=2,
                      source=source,
                      color=color,
                      legend=f"{agent.address} Balance")

        plot.legend.location = 'top_left'
        plot.legend.label_text_font_size = '12pt'

        plot.xaxis[0].formatter = FuncTickFormatter(code="""
        return (tick / 86400).toFixed(0);
        """)
        plot.yaxis[0].formatter = PrintfTickFormatter(format="%0.1f%%")

        acc_source = ColumnDataSource(dict(t=[], a=[]))
        if baseline_accuracy is not None:
            plot.ray(x=[0], y=[baseline_accuracy * 100], length=0, angle=0, line_width=2,
                     legend=f"Accuracy when trained with all data: {baseline_accuracy * 100:0.1f}%")
            
        plot.line(x='t', y='a',
                  line_dash='solid',
                  line_width=2,
                  source=acc_source,
                  color='black',
                  legend="Current Accuracy")

        @gen.coroutine
        def plot_cb(agent: Agent, t, b):
            source = balance_plot_sources_per_agent[agent.address]
            source.stream(dict(t=[t], b=[b * 100 / agent.start_balance]))
            save_data['balances'].append(dict(t=t, a=agent.address, b=b))

        @gen.coroutine
        def plot_accuracy_cb(t, a):
            acc_source.stream(dict(t=[t], a=[a * 100]))
            save_data['accuracies'].append(dict(t=t, accuracy=a))

        continuous_evaluation = not isinstance(self._simu.im, PredictionMarket)

        def task():
            (x_train, y_train), (x_test, y_test) = \
                self._data_loader.load_data(
                    train_size=train_size, test_size=test_size)
            classifications = self._data_loader.classifications()
            x_train, x_test, feature_index_mapping = self._feature_index_mapper.map(
                x_train, x_test)
            x_train_len = x_train.shape[0]
            init_idx = int(x_train_len * init_train_data_portion)
            self._logger.info("Initializing model with %d out of %d samples.",
                              init_idx, x_train_len)
            x_init_data, y_init_data = x_train[:init_idx], y_train[:init_idx]
            x_remaining, y_remaining = x_train[init_idx:], y_train[init_idx:]

            save_model = isinstance(
                self._simu.im, PredictionMarket) and self._simu.im.reset_model_during_reward_phase
            self._simu.model.init_model(x_init_data, y_init_data, save_model)

            if self._logger.isEnabledFor(logging.DEBUG):
                s = self._simu.model.evaluate(x_init_data, y_init_data)
                self._logger.debug("Initial training data evaluation: %s", s)
                if len(x_remaining) > 0:
                    s = self._simu.model.evaluate(x_remaining, y_remaining)
                    self._logger.debug(
                        "Remaining training data evaluation: %s", s)
                else:
                    self._logger.debug(
                        "There is no more remaining data to evaluate.")

            self._logger.info("Evaluating initial model.")
            accuracy = self._simu.model.log_evaluation_details(x_test, y_test)
            self._logger.info(
                "Initial test set accuracy: %0.2f%%", accuracy * 100)
            t = self._time()
            doc.add_next_tick_callback(
                partial(plot_accuracy_cb, t=t, a=accuracy))

            q = PriorityQueue()
            random.shuffle(agents)
            for agent in agents:
                self._balances.initialize(agent.address, agent.start_balance)
                q.put((self._time() + agent.get_next_wait_s(), agent))
                doc.add_next_tick_callback(
                    partial(plot_cb, agent=agent, t=t, b=agent.start_balance))

            unclaimed_data = []
            next_data_index = 0
            next_accuracy_plot_time = 1E4
            desc = "Processing agent requests"
            current_time = 0
            with tqdm(desc=desc,
                      unit_scale=True, mininterval=2, unit=" requests",
                      total=len(x_remaining),
                      ) as pbar:
                while not q.empty():
                    if next_data_index >= len(x_remaining):
                        if not continuous_evaluation or len(unclaimed_data) == 0:
                            break

                    current_time, agent = q.get()
                    update_balance_plot = False
                    if current_time > next_accuracy_plot_time:
                        self._logger.debug("Evaluating.")
                        next_accuracy_plot_time += accuracy_plot_wait_s
                        accuracy = self._simu.model.evaluate(x_test, y_test)
                        doc.add_next_tick_callback(
                            partial(plot_accuracy_cb, t=current_time, a=accuracy))

                        if continuous_evaluation:
                            self._logger.debug(
                                "Unclaimed data: %d", len(unclaimed_data))
                            pbar.set_description(
                                f"{desc} ({len(unclaimed_data)} unclaimed)")

                        with open(save_path, 'w') as f:
                            json.dump(save_data, f, separators=(',', ':'))
                        self._simu.model.export(model_save_path, classifications,
                                                feature_index_mapping=feature_index_mapping)

                        if os.path.exists(plot_save_path):
                            os.remove(plot_save_path)
                        self.save_plot_image(plot, plot_save_path)

                    self._time.set_time(current_time)

                    balance = self._balances[agent.address]
                    if balance > 0 and next_data_index < len(x_remaining):
                        # Pick data.
                        x, y = x_remaining[next_data_index], y_remaining[next_data_index]

                        if agent.calls_model:
                            # Only call the model if it's good.
                            if random.random() < accuracy:
                                update_balance_plot = True
                                self._simu.predict(
                                    Msg(agent.address, agent.pay_to_call), x)
                        else:
                            if not agent.good:
                                y = 1 - y
                            if agent.prob_mistake > 0 and random.random() < agent.prob_mistake:
                                y = 1 - y

                            # Bad agents always contribute.
                            # Good agents will only work if the model is doing well.
                            # Add a bit of chance they will contribute since 0.85 accuracy is okay.
                            if not agent.good or random.random() < accuracy + 0.15:
                                value = agent.get_next_deposit()
                                if value > balance:
                                    value = balance
                                msg = Msg(agent.address, value)
                                try:
                                    self._simu.add_data(msg, x, y)
                                    # only plot data in particular period
                                    update_balance_plot = next_data_index / \
                                        len(x_remaining) + \
                                        0.1 < random.random()
                                    balance = self._balances[agent.address]
                                    if continuous_evaluation:
                                        unclaimed_data.append(
                                            (current_time, agent, x, y))
                                    next_data_index += 1
                                    pbar.update()
                                except RejectException:
                                    if self._logger.isEnabledFor(logging.DEBUG):
                                        self._logger.exception(
                                            "Error adding data.")

                    if balance > 0:
                        q.put((current_time + agent.get_next_wait_s(), agent))

                    claimed_indices = []
                    for i in range(len(unclaimed_data)):
                        added_time, adding_agent, x, classification = unclaimed_data[i]
                        if current_time - added_time < self._simu.im.refund_time_s:
                            break
                        if next_data_index >= len(x_remaining) \
                                and current_time - added_time < self._simu.im.any_address_claim_wait_time_s:
                            break
                        balance = self._balances[agent.address]
                        msg = Msg(agent.address, balance)

                        if current_time - added_time > self._simu.im.any_address_claim_wait_time_s:
                            # Attempt to take the entire deposit.
                            try:
                                self._simu.report(
                                    msg, x, classification, added_time, adding_agent.address)
                                update_balance_plot = True
                            except RejectException:
                                if self._logger.isEnabledFor(logging.DEBUG):
                                    self._logger.exception(
                                        "Error taking reward.")
                        elif adding_agent.address == agent.address:
                            try:
                                self._simu.refund(
                                    msg, x, classification, added_time)
                                update_balance_plot = True
                            except RejectException:
                                if self._logger.isEnabledFor(logging.DEBUG):
                                    self._logger.exception(
                                        "Error getting refund.")
                        else:
                            try:
                                self._simu.report(
                                    msg, x, classification, added_time, adding_agent.address)
                                update_balance_plot = True
                            except RejectException:
                                if self._logger.isEnabledFor(logging.DEBUG):
                                    self._logger.exception(
                                        "Error taking reward.")

                        stored_data = self._simu.data_handler.get_data(x, classification,
                                                                       added_time, adding_agent.address)
                        if stored_data.claimable_amount <= 0:
                            claimed_indices.append(i)

                    for i in claimed_indices[::-1]:
                        unclaimed_data.pop(i)

                    if update_balance_plot:
                        balance = self._balances[agent.address]
                        doc.add_next_tick_callback(
                            partial(plot_cb, agent=agent, t=current_time, b=balance))

            self._logger.info("Done going through data.")
            if continuous_evaluation:
                pbar.set_description(
                    f"{desc} ({len(unclaimed_data)} unclaimed)")

            if isinstance(self._simu.im, PredictionMarket):
                self._time.add_time(agents[0].get_next_wait_s())
                self._simu.im.end_market()
                for i, test_set_portion in enumerate(pm_test_sets):
                    if i != self._simu.im.test_reveal_index:
                        self._simu.im.verify_next_test_set(test_set_portion)
                with tqdm(desc="Processing contributions",
                          unit_scale=True, mininterval=2, unit=" contributions",
                          total=self._simu.im.get_num_contributions_in_market(),
                          ) as pbar:
                    finished_first_round_of_rewards = False
                    while self._simu.im.remaining_bounty_rounds > 0:
                        self._time.add_time(agents[0].get_next_wait_s())
                        self._simu.im.process_contribution()
                        pbar.update()

                        if not finished_first_round_of_rewards:
                            accuracy = self._simu.im.prev_acc
                            if random.random() < 0.1:
                                doc.add_next_tick_callback(
                                    partial(plot_accuracy_cb, t=self._time(), a=accuracy))

                        if self._simu.im.state == MarketPhase.REWARD_RESTART:
                            finished_first_round_of_rewards = True
                            if self._simu.im.reset_model_during_reward_phase:
                                # Update the accuracy after resetting all data.
                                accuracy = self._simu.im.prev_acc
                            else:
                                pass
                            doc.add_next_tick_callback(
                                partial(plot_accuracy_cb, t=self._time(), a=accuracy))
                            pbar.total += self._simu.im.get_num_contributions_in_market()
                            self._time.add_time(self._time() * 0.001)

                            for agent in agents:
                                balance = self._balances[agent.address]
                                market_bal = self._simu.im._market_balances[agent.address]
                                self._logger.debug("\"%s\" market balance: %0.2f   Balance: %0.2f",
                                                   agent.address, market_bal, balance)
                                doc.add_next_tick_callback(
                                    partial(plot_cb, agent=agent, t=self._time(), b=max(balance + market_bal, 0)))

                self._time.add_time(self._time() * 0.02)
                for agent in agents:
                    msg = Msg(agent.address, 0)
                    data = None
                    for key, stored_data in self._simu.data_handler:
                        if stored_data.sender == agent.address:
                            data = key[0]
                            break
                    if data is not None:
                        self._simu.refund(msg, np.array(
                            data), stored_data.classification, stored_data.time)
                        balance = self._balances[agent.address]
                        doc.add_next_tick_callback(
                            partial(plot_cb, agent=agent, t=self._time(), b=balance))
                        self._logger.info("Balance for \"%s\": %.2f (%+.2f%%)",
                                          agent.address, balance,
                                          (balance - agent.start_balance) / agent.start_balance * 100)
                    else:
                        self._logger.warning("No data submitted by \"%s\" was found."
                                             "\nWill not update it's balance.", agent.address)

                self._logger.info("Done issuing rewards.")

            accuracy = self._simu.model.log_evaluation_details(x_test, y_test)
            doc.add_next_tick_callback(
                partial(plot_accuracy_cb, t=current_time + 100, a=accuracy))

            with open(save_path, 'w') as f:
                json.dump(save_data, f, separators=(',', ':'))
            self._simu.model.export(
                model_save_path, classifications, feature_index_mapping=feature_index_mapping)

            if os.path.exists(plot_save_path):
                os.remove(plot_save_path)
            self.save_plot_image(plot, plot_save_path)

        doc.add_root(plot)
        thread = Thread(target=task)
        thread.start()
