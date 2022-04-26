import math
from collections import Counter
from logging import Logger
from injector import inject, Module, singleton

from simu.simulation.contract.balances import Balances
from simu.simulation.contract.data.data_handler import StoredData
from simu.simulation.contract.incentive.incentive_mechanism import IncentiveMechanism
from simu.simulation.contract.objects import Address, RejectException, TimeMock

"""
    adding data needs deposit
    good agent can reclaim deposit and get reward.
"""
@singleton
class Stakeable(IncentiveMechanism):

    @inject
    def __init__(self,
                 # Injected
                 balances: Balances,logger: Logger,time_method: TimeMock,
                 # Parameters
                 refund_time_s=60 * 60 * 24 * 1,any_address_claim_wait_time_s=60 * 60 * 24 * 9,cost_weight=1,):
        super().__init__(refund_time_s=refund_time_s,any_address_claim_wait_time_s=any_address_claim_wait_time_s)

        self._balances = balances
        self._logger = logger
        self._time = time_method

        # refund will happend at least one week
        min_refund_window_s = 60 * 60 * 24 * 7
        assert self.any_address_claim_wait_time_s > self.refund_time_s + min_refund_window_s, "Claim is not ready, please wait..."

        self.cost_weight = cost_weight

        self.num_good_data_per_user = Counter()
        self.total_num_good_data = 0
        self._last_update_time_s = int(self._time())

    def distribute_payment_for_prediction(self, sender, value):
        if value > 0:
            for agent_address, num_good in self.num_good_data_per_user.items():
                # total value distributed < value.
                self._balances.send(sender, agent_address, int(value * num_good / self.total_num_good_data))

    """
        :param data: A single sample of training data for the model.
        :param classification: The labels for `data`.
        :return: The current cost to update a model with a specific sample of training data.
    """
    def get_cost_for_next_add_data(self, data, classification) -> float:
        
        current_time_s = int(self._time())
        time_since_last_update_s = int((current_time_s - self._last_update_time_s))
        
        if time_since_last_update_s <= 0:
            raise RejectException("Not enough time has passed since the last update.")

        # cost = self.cost_weight / int(math.sqrt(time_since_last_update_s * 3600))
        result = self.cost_weight * 60 / int(math.sqrt(time_since_last_update_s))
        result = int(result)
        
        # minimum cost is 1
        if result < 1:
            result = 1
        return result

    def handle_add_data(self, contributor_address: Address, msg_value: float, data, classification) -> (float, bool):
        cost = self.get_cost_for_next_add_data(data, classification)
        update_model = True
        if cost > msg_value:
            raise RejectException(
                f"Did not pay enough. Sent {msg_value} < {cost}")
        self._last_update_time_s = self._time()
        return (cost, update_model)

    def handle_refund(self, submitter: str, stored_data: StoredData,
                      claimable_amount: float, claimed_by_submitter: bool,
                      prediction) -> float:
        result = claimable_amount

        if claimed_by_submitter:
            raise RejectException("Deposit already claimed")
        if result <= 0:
            raise RejectException("Reward already claimed")
        
        current_time_s = int(self._time())
        if current_time_s - stored_data.time <= self.refund_time_s:
            raise RejectException("Wait refund to be handled")
        if callable(prediction):
            prediction = prediction()
        if prediction != stored_data.classification:
            raise RejectException("The model doesn't agree with your contribution")

        self.num_good_data_per_user[submitter] += 1
        self.total_num_good_data += 1

        return result

    def handle_report(self, reporter: str, stored_data: StoredData, claimed_by_reporter: bool, prediction) -> float:
        if stored_data.claimable_amount <= 0:
            raise RejectException("Reward already claimed")

        current_time_s = int(self._time())
        if current_time_s - stored_data.time >= self.any_address_claim_wait_time_s:
            # Enough time has passed, give the entire remaining deposit to the reporter.
            self._logger.debug("Giving all remaining deposit to \"%s\".", reporter)
            result = stored_data.claimable_amount
            return result

        # bad agent cannot claim back their own deposit
        if reporter == stored_data.sender:
            raise RejectException("Cannot take your own deposit. Ask for a refund instead.")
        if claimed_by_reporter:
            raise RejectException("Deposit already claimed")
        if current_time_s - stored_data.time <= self.refund_time_s:
            raise RejectException("ait refund to be handled")
        if callable(prediction):
            prediction = prediction()
        if prediction == stored_data.classification:
            raise RejectException("The model should not agree with the contribution.")

        num_good = self.num_good_data_per_user[reporter]
        if num_good <= 0:
            raise RejectException(f"No good data was verified by reporter '{reporter}'.")
        result = stored_data.initial_deposit * num_good / self.total_num_good_data
        
        # Handle possible rounding errors or if there is too little to divide to reporters.
        if result <= 0 or result > stored_data.claimable_amount:
            result = stored_data.claimable_amount
        return result


class StakeableImModule(Module):
    def configure(self, binder):
        binder.bind(IncentiveMechanism, to=Stakeable)
