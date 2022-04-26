from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from injector import inject, singleton

from simu.simulation.contract.objects import Address, RejectException, SmartContract, TimeMock


@dataclass
class StoredData:
    # Storing the data is not necessary. data: object
    classification: object
    time: int
    sender: Address

    # initial deposit for the data
    initial_deposit: float

    # deposit can be claimed
    claimable_amount: float

    claimed_by: Dict[Address, bool] = field(
        default_factory=lambda: defaultdict(bool))

"""
    Stores added training data and meta-data.
"""
@inject
@singleton
@dataclass
class DataHandler(SmartContract):
    _time: TimeMock

    _added_data: Dict[tuple, StoredData] = field(
        default_factory=dict, init=False)

    def __iter__(self):
        return iter(self._added_data.items())

    def _get_key(self, data, classification, added_time: int, original_author: Address):
        if isinstance(data, np.ndarray):
            data = tuple(data.tolist())
        else:
            data = tuple(data)
        return (data, classification, added_time, original_author)

    """
        :param data: original data
        :param classification: original label
        :param added_time
        :param original_author
        :return: data info
    """
    def get_data(self, data, classification, added_time: int, original_author: Address) -> StoredData:
        key = self._get_key(data, classification, added_time, original_author)
        result = self._added_data.get(key)
        return result

    """
        :param sender: address of agent
        :param cost
        :param data: sample of training data for the model.
        :param classification: label
    """
    def handle_add_data(self, contributor_address: Address, cost, data, classification):
        current_time_s = self._time()
        key = self._get_key(data, classification,
                            current_time_s, contributor_address)
        if key in self._added_data:
            raise RejectException("Data has already been added.")
        d = StoredData(classification, current_time_s,
                       contributor_address, cost, cost)
        self._added_data[key] = d

    """
        :param submitter: The address of the agent attempting a refund
        :param data: The data for which to attempt a refund
        :param classification: original label
        :param added_time
        :return: refund amount
    """
    def handle_refund(self, submitter: Address, data, classification, added_time: int) -> (float, bool, StoredData):
        stored_data = self.get_data(
            data, classification, added_time, submitter)
        assert stored_data is not None, "Data not found."
        assert stored_data.sender == submitter, "Data isn't from the sender."
        claimable_amount = stored_data.claimable_amount
        claimed_by_submitter = stored_data.claimed_by[submitter]

        return (claimable_amount, claimed_by_submitter, stored_data)

    """
        :param reporter: address of the agent reporting the data
        :param data: report data
        :param classification: original label if data
        :param added_time
        :param original_author: address that originally added the data
        :return: data
    """
    def handle_report(self, reporter: Address, data, classification, added_time: int, original_author: Address) \
            -> (bool, StoredData):
        stored_data = self.get_data(
            data, classification, added_time, original_author)
        assert stored_data is not None, "Data not found."
        claimed_by_reporter = stored_data.claimed_by[reporter]

        return (claimed_by_reporter, stored_data)

    def update_claimable_amount(self, receiver: Address, stored_data: StoredData, reward_amount: float):
        if reward_amount > 0:
            stored_data.claimed_by[receiver] = True
            stored_data.claimable_amount -= reward_amount
