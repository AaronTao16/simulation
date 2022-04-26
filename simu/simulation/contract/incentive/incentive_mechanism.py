import math
from abc import ABC, abstractmethod

from simu.simulation.contract.data.data_handler import StoredData
from simu.simulation.contract.objects import Address, SmartContract

"""
    incentive mechanism
"""
class IncentiveMechanism(ABC, SmartContract):
    def __init__(self, refund_time_s=math.inf, any_address_claim_wait_time_s=math.inf):
        super().__init__()
        # time to wait for refund
        self.refund_time_s = refund_time_s

        # time to wait for others getting deposit
        self.any_address_claim_wait_time_s = any_address_claim_wait_time_s

    @abstractmethod
    def distribute_payment_for_prediction(self, sender: str, value: float):
        """
        Share `value` with those that submit data.

        :param sender: The address of the one calling prediction.
        :param value: The amount sent with the request to call prediction.
        """
        pass

    """
        Determine if the request to add data is acceptable.

        :param contributor_address: The address of the one attempting to add data
        :param msg_value: The value sent with the initial transaction to add data.
        :param data: A single sample of training data for the model.
        :param classification: The label for `data`.
        :return: tuple
            The cost required to add new data.
            `True` if the model should be updated, `False` otherwise.
    """
    @abstractmethod
    def handle_add_data(self, contributor_address: Address, msg_value: float, data, classification) \
            -> (float, bool):
        pass

    """
        Notify that a refund is being attempted.

        :param submitter: The address of the one attempting a refund.
        :param stored_data: The data for which a refund is being attempted.
        :param claimable_amount: The amount that can be claimed for the refund.
        :param claimed_by_submitter: True if the data has already been claimed by `submitter`, otherwise false.
        :param prediction: The current prediction of the model for data
            or a callable with no parameters to lazily get the prediction of the model on the data.
        :return: The amount to refund to `submitter`.
    """
    @abstractmethod
    def handle_refund(self, submitter: str, stored_data: StoredData,
                      claimable_amount: float, claimed_by_submitter: bool,
                      prediction) -> float:
        pass

    """
        Notify that data is being reported as bad or old.

        :param reporter: The address of the one reporting about the data.
        :param stored_data: The data being reported.
        :param claimed_by_reporter: True if the data has already been claimed by `reporter`, otherwise false.
        :param prediction: The current prediction of the model for data
        :return: reward to reporter
    """
    @abstractmethod
    def handle_report(self, reporter: str, stored_data: StoredData, claimed_by_reporter: bool, prediction) \
            -> float:
        pass
