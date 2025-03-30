from collections import defaultdict
from itertools import combinations
from graphviz import Digraph
import pandas as pd


class TreeNode:
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # Связь для одноименных узлов

    def increment(self, count):
        self.count += count


class FPGrowth:
    def __init__(self, df, min_support):
        self.df = df
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.rules = []

    def build_fptree(self):
        item_support = defaultdict(int)
        for _, transaction in self.df.iterrows():
            for item, present in transaction.items():
                if present:
                    item_support[item] += 1

        # Отфильтровываем элементы с низкой поддержкой
        item_support = {item: count for item, count in item_support.items() if count / len(self.df) >= self.min_support}
        if not item_support:
            return None, None

        print(f'item_support длиной {len(item_support)}: {item_support}')

        def sort_transaction(transaction):
            return [item for item, present in sorted(transaction.items(), key=lambda x: item_support.get(x[0], 0),
                                                     reverse=True) if present and item in item_support]

        root = TreeNode('null', 1, None)
        header_table = defaultdict(list)
        k = 1
        for _, transaction in self.df.iterrows():
            sorted_items = sort_transaction(transaction)
            print(f'{k}. sorted_items: {sorted_items}')
            k += 1
            if sorted_items:
                self._insert_tree(sorted_items, root, header_table)

        return root, header_table

    def _insert_tree(self, items, node, header_table):
        first_item = items[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = TreeNode(first_item, 1, node)
            node.children[first_item] = new_node
            header_table[first_item].append(new_node)

        if len(items) > 1:
            self._insert_tree(items[1:], node.children[first_item], header_table)

    # def visualize_tree_graph(self, node, graph=None):
    #     # Визуализация FP-дерева
    #     if graph is None:
    #         graph = Digraph(format='jpeg')
    #         graph.attr('node', shape='circle')  # Узлы в форме круга
    #
    #     # Заменяем пробелы в имени узла на подчеркивания
    #     node_label = f"{node.name}: {node.count}".replace(' ', '_')
    #     graph.node(node_label)  # Добавляем узел
    #
    #     for child in node.children.values():
    #         # Аналогично заменяем пробелы в дочерних узлах
    #         child_label = f"{child.name}: {child.count}".replace(' ', '_')
    #         graph.node(child_label)  # Добавляем дочерний узел
    #         graph.edge(node_label, child_label)  # Связываем родителя и потомка
    #
    #         # Рекурсия для обработки дочерних узлов
    #         self.visualize_tree_graph(child, graph)
    #
    #     return graph
    #
    # def save_tree_as_jpeg(self, root):
    #     # Сохранение дерева в формате JPEG
    #     graph = self.visualize_tree_graph(root)
    #     graph.render('fp_tree')  # Сохраняем в файл

    def _mine_tree(self, header_table, prefix):
        for base in sorted(header_table, key=lambda x: len(header_table[x])):
            new_freq_set = prefix.copy()
            new_freq_set.add(base)
            support = sum(node.count for node in header_table[base]) / len(self.df)
            if support >= self.min_support:
                self.frequent_itemsets[frozenset(new_freq_set)] = support
                conditional_base = self._find_conditional_base(base, header_table)
                conditional_tree, conditional_header = self._build_conditional_tree(conditional_base)
                if conditional_header:
                    self._mine_tree(conditional_header, new_freq_set)

    def _find_conditional_base(self, base, header_table):
        conditional_base = []
        for node in header_table[base]:
            path = []
            parent = node.parent
            while parent and parent.name != 'null':
                path.append(parent.name)
                parent = parent.parent
            if path:
                conditional_base.extend([path] * node.count)
        return conditional_base

    def _build_conditional_tree(self, conditional_base):
        item_support = defaultdict(int)
        for path in conditional_base:
            for item in path:
                item_support[item] += 1

        item_support = {item: count for item, count in item_support.items() if count / len(self.df) >= self.min_support}
        if not item_support:
            return None, None

        root = TreeNode('null', 1, None)
        header_table = defaultdict(list)
        for path in conditional_base:
            sorted_path = [item for item in path if item in item_support]
            if sorted_path:
                self._insert_tree(sorted_path, root, header_table)

        return root, header_table

    def find_frequent_itemsets(self):
        root, header_table = self.build_fptree()
        if root and header_table:
            self._mine_tree(header_table, set())

    def generate_association_rules(self, min_confidence):
        for itemset in self.frequent_itemsets:
            if len(itemset) > 1:
                for subset in self._get_subsets(itemset):
                    antecedent = frozenset(subset)
                    consequent = itemset - antecedent
                    antecedent_support = self.frequent_itemsets.get(antecedent, 0)
                    itemset_support = self.frequent_itemsets[itemset]
                    confidence = itemset_support / antecedent_support if antecedent_support else 0

                    if confidence >= min_confidence:
                        self.rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': itemset_support,
                            'confidence': confidence
                        })

    def _get_subsets(self, itemset):
        return [set(comb) for i in range(1, len(itemset)) for comb in combinations(itemset, i)]

    def display_frequent_itemsets(self):
        print("Frequent Itemsets:")
        k = 1
        for itemset, support in self.frequent_itemsets.items():
            print(f"{k}. {set(itemset)}: {support:.2f}")
            k += 1

    def display_rules(self):
        print("\nAssociation Rules:")
        k = 1
        for rule in self.rules:
            antecedent = ", ".join(rule['antecedent'])
            consequent = ", ".join(rule['consequent'])
            support = rule['support']
            confidence = rule['confidence']
            print(f"Rule ({k}): {antecedent} -> {consequent}, Support: {support:.2f}, Confidence: {confidence:.2f}")
            k += 1


df = pd.read_csv('../df_onehot.csv')
df.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)

fpgrowth = FPGrowth(df, 0.1)
fpgrowth.find_frequent_itemsets()
fpgrowth.generate_association_rules(0.6)
fpgrowth.display_frequent_itemsets()
fpgrowth.display_rules()