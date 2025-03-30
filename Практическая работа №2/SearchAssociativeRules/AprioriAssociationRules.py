import pandas as pd
from itertools import combinations


class AprioriAssociationRules:
    def __init__(self, data, min_support):
        """
        Инициализация класса
        :param data: DataFrame с транзакциями (булевые значения для товаров)
        :param min_support: Минимальная поддержка для частых наборов
        """
        self.df = data
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.rules = []

    def _get_support(self, itemset):
        """
        Вычислить поддержку для данного набора товаров
        :param itemset: Набор товаров
        :return: Поддержка (доля транзакций, содержащих все товары в наборе)
        """
        mask = self.df[list(itemset)].all(axis=1)
        support = mask.mean()
        return support

    def find_frequent_itemsets(self):
        """
        Поиск всех частых наборов элементов с учетом минимальной поддержки
        """
        items = self.df.columns
        current_itemsets = []
        for item in items:
            current_itemsets.append({item})
        k = 1

        while current_itemsets:
            frequent_itemsets_k = {}

            for itemset in current_itemsets:
                support = self._get_support(itemset)
                if support >= self.min_support:
                    frequent_itemsets_k[frozenset(itemset)] = support

            if not frequent_itemsets_k:
                break

            self.frequent_itemsets.update(frequent_itemsets_k)

            new_itemsets = []
            for i in frequent_itemsets_k.keys():
                for j in frequent_itemsets_k.keys():
                    new_itemset = i.union(j)
                    if len(new_itemset) == k + 1 and new_itemset not in new_itemsets:
                        new_itemsets.append(new_itemset)

            current_itemsets = new_itemsets
            k += 1

    def generate_association_rules(self, min_confidence):
        """
        Генерация ассоциативных правил на основе частых наборов
        :param min_confidence: Минимальная уверенность (confidence) для правил
        """
        for itemset in self.frequent_itemsets:
            if len(itemset) > 1:
                for subset in self._get_subsets(itemset):
                    if subset:
                        antecedent = frozenset(subset)
                        consequent = itemset - antecedent
                        antecedent_support = self.frequent_itemsets[antecedent]
                        itemset_support = self.frequent_itemsets[itemset]

                        # confidence
                        confidence = itemset_support / antecedent_support

                        # lift
                        consequent_support = self.frequent_itemsets[frozenset(consequent)]
                        lift = confidence / consequent_support if consequent_support > 0 else 0

                        # leverage
                        leverage = itemset_support - (antecedent_support * consequent_support)

                        # conviction
                        antecedent_count = antecedent_support
                        consequent_count = consequent_support
                        confidence_count = itemset_support
                        conviction = (antecedent_count * (1 - consequent_count)) / (
                                antecedent_count - confidence_count) if antecedent_count - confidence_count > 0 else 0

                        if confidence >= min_confidence:
                            rule = {
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': itemset_support,
                                'confidence': confidence,
                                'lift': lift,
                                'leverage': leverage,
                                'conviction': conviction
                            }
                            self.rules.append(rule)

    def _get_subsets(self, itemset):
        """
        Генерация всех подмножеств для данного набора элементов
        :param itemset: Набор элементов
        :return: Подмножества набора элементов
        """
        subsets = []
        for i in range(1, len(itemset)):
            for comb in combinations(itemset, i):
                subsets.append(set(comb))
        return subsets

    def display_frequent_itemsets(self):
        """
        Показать частые наборы элементов и их поддержку
        """
        k = 1
        print("Frequent Itemsets:")
        for itemset, support in self.frequent_itemsets.items():
            print(f"{k}. {set(itemset)}: {support:.2f}")
            k += 1

    def display_rules(self):
        """
        Показать ассоциативные правила
        """
        print("\nAssociation Rules:")
        k = 1
        for rule in self.rules:
            antecedent = ", ".join(rule['antecedent'])
            consequent = ", ".join(rule['consequent'])
            support = rule['support']
            confidence = rule['confidence']
            lift = rule['lift']
            leverage = rule['leverage']
            conviction = rule['conviction']
            print(
                f"Rule ({k}): {antecedent} -> {consequent}, Support: {support:.2f}, Confidence: {confidence:.2f}, "
                f"Lift: {lift:.2f}, Leverage: {leverage:.2f}, Conviction: {conviction:.2f}")
            print(rule)
            k += 1
