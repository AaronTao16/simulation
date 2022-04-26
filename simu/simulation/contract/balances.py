from dataclasses import dataclass, field
from logging import Logger
from typing import Dict

from injector import inject, singleton

from simu.simulation.contract.objects import Address

# agent balance
@inject
@singleton
@dataclass
class Balances(object):
    _logger: Logger

    _balances: Dict[Address, float] = field(default_factory=dict, init=False)

    """
        :param address: agent's address
        :return: `True` if the address exists
    """
    def __contains__(self, address: Address):
        return address in self._balances


    """
        :param address: agent's address
        :return: balance
    """
    def __getitem__(self, address: Address) -> float:
        return self._balances[address]

    def get_all(self) -> Dict[Address, float]:
        return dict(self._balances)

    """ Initialize agent's balance. """
    def initialize(self, address: Address, start_balance: float):    
        assert address not in self._balances, f"'{address}' already has a balance."
        self._balances[address] = start_balance

    """ Send funds from one agent to another. """
    def send(self, sending_address: Address, receiving_address: Address, amount):
        assert amount >= 0
        if amount > 0:
            sender_balance = self._balances[sending_address]
            if sender_balance < amount:
                self._logger.warning(f"'{sending_address} has {sender_balance} < {amount}.\n"
                                     f"Will only send {sender_balance}.")
                amount = sender_balance

            self._balances[sending_address] -= amount
            if receiving_address not in self._balances:
                self.initialize(receiving_address, amount)
            else:
                self._balances[receiving_address] += amount
