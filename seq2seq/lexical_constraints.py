import copy
import logging
from operator import attrgetter
from typing import Dict, List, Optional, Tuple, Set, Union
import collections
import stanza
import numpy as np
import torch
import predict_dtree

logger = logging.getLogger(__name__)

Phrase = List[int]
Literal = Tuple[Phrase, bool]
# Represents a list of raw constraints for a sentence. Each constraint is a list of target-word IDs.
RawConstraintList = List[Phrase]
ClauseConstraintList = List[List[Literal]]
nlp = stanza.Pipeline('en', processors='lemma,tokenize,pos,depparse')


class Pair:
    
    def __init__(self,
                 raw_phrases: Optional[RawConstraintList] = None):
        self.i = 0
        self.pairs =[]
        if raw_phrases:
            for phrase in raw_phrases:
                self.add_phrase(phrase)
    def add_phrase(self,phrase):
        self.pairs.append({'word':phrase[0],'role':phrase[1:]})
        
    def __str__(self):
        # s = f'({list(self.final_ids)}'
        s ='('
        for item in self.pairs:
            # s += f' -> {child_id} {self.children[child_id]}'
            s += f" {item['word']} : {item['role']},"
        s += ')'
        return s
    def __iter__(self):
        return self.pairs

    def __next__(self):
        x = self.pairs[self.i]
        self.i += 1
        return x
    def get_item(self,ind):
        return self.pairs[ind]
    
    def __len__(self):
        return len(self.pairs)
        
                    
# TODO needs to be fixed
class NegativeState:
    """
    Repesrents the state of a hypothesis in the AvoidTrie.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param avoid_trie: The trie containing the phrases to avoid.
    :param state: The current state (defaults to root).
    """
    def __init__(self,
                 avoid_pair: Pair,
                ):

        self.root = avoid_pair

    def consume(self, sentence: str):#word_id: int):
        """
        Consumes a word, and updates the state based on it. Returns new objects on a state change.

        The next state for a word can be tricky. Here are the cases:
        (1) If the word is found in our set of outgoing child arcs, we take that transition.
        (2) If the word is not found, and we are not in the root state, we need to reset.
            This means we pretend we were in the root state, and see if we can take a step
        (3) Otherwise, if we are not already in the root state (i.e., we were partially through
            the trie), we need to create a new object whose state is the root state
        (4) Finally, if we couldn't advance and were already in the root state, we can reuse
            this object.

        :param word_id: The word that was just generated.
        """
        return self
        # new_state = []
        # for state in set(self.state + [self.root]):
        #     if word_id in state.children:
        #         new_state.append(state.step(word_id))

        # if new_state:
        #     return NegativeState(self.root)
        # else:
        #     if len(self.state) == 1 and self.root == self.state[0]:
        #         return self
        #     else:
        #         return NegativeState(self.root)

    def avoid(self):
        """
        Returns a set of word IDs that should be avoided. This includes the set of final states from the
        root node, which are single tokens that must never be generated.

        :return: A set of integers representing words that must not be generated next by this hypothesis.
        """
        
        return set()
        # return self.root.final().union(*[state.final() for state in self.state])

    def __str__(self):
        return str(self.root)

## TODO fix it, it shouldn't work
class NegativeBatch:
    """
    Represents a set of phrasal constraints for all items in the batch.
    For each hypotheses, there is an AvoidTrie tracking its state.

    :param beam_size: The beam size.
    :param avoid_list: The list of lists (raw phrasal constraints as IDs, one for each item in the batch).
    """
    def __init__(self,
                 beam_size: int,
                 avoid_list: Optional[List[RawConstraintList]] = None):

        self.avoid_states = []  # type: List[NegativeState]

        # Store the sentence-level tries for each item in their portions of the beam
        if avoid_list is not None:
            for literal_phrases in avoid_list:
                self.avoid_states += [NegativeState(Pair(literal_phrases))] * beam_size

    def reorder(self, indices: torch.Tensor):
        """
        Reorders the avoid list according to the selected row indices.
        This can produce duplicates, but this is fixed if state changes occur in consume().

        :param indices: An mx.nd.NDArray containing indices of hypotheses to select.
        """
        if self.avoid_states:
            self.avoid_states = [self.avoid_states[x] for x in indices.numpy()]

    def consume(self, word_ids: torch.Tensor):
        """
        Consumes a word for each trie, updating respective states.

        :param word_ids: The set of word IDs.
        """
        word_ids = word_ids.numpy().tolist()
        for i, word_id in enumerate(word_ids):
            if self.avoid_states:
                self.avoid_states[i] = self.avoid_states[i].consume(word_id)

    def avoid(self) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Assembles a list of per-hypothesis words to avoid. The indices are (x, y) pairs into the scores
        array, which has dimensions (beam_size, target_vocab_size). These values are then used by the caller
        to set these items to np.inf so they won't be selected. Words to be avoided are selected by
        consulting both the global trie of phrases and the sentence-specific one.

        :return: Two lists of indices: the x coordinates and y coordinates.
        """
        to_avoid = set()  # type: Set[Tuple[int, int]]
        for i, state in enumerate(self.avoid_states):
            for word_id in state.avoid():
                to_avoid.add((i, word_id))

        return tuple(zip(*to_avoid))  # type: ignore

class PositiveState:
    """
    Represents a set of words and phrases that must appear in the output.
    The offset is used to return actual positions in the one-dimensionally-resized array that
    get set to infinity.

    :param positive_trie: The trie containing the phrases to appear.
    :param state: The current state (defaults to root).
    """
    # def __init__(self,
    #              positive_trie: Trie,
    #              state: List[Trie] = None,
    #              met_phrases: RawConstraintList = None) -> None:
    def __init__(self,
                 positive_pair: Pair,
                 met_phrases: RawConstraintList = None):  
        self.root = positive_pair
        self.met_phrases = met_phrases

    def __str__(self):
        s = f'Root: ['
        for i in range(len(self.root)):
            s += f'{self.root.get_item(i)}, '
        s += f']\nMet_phrases: {self.met_phrases}'
        return s

    def allowed(self):
        """
        Returns the set of constrained words that could follow this one.
        For unfinished phrasal constraints, it is the next word in the phrase.
        In other cases, it is the list of all unmet constraints.
        If all constraints are met, an empty set is returned.

        :return: The ID of the next required word, or -1 if any word can follow
        """

        return set()

    def advance(self, sentence: str):
        """
        Updates the constraints object based on advancing on word_id.
        There is a complication, in that we may have started but not
        yet completed a multi-word constraint.  We need to allow constraints
        to be added as unconstrained words, so if the next word is
        invalid, we must "back out" of the current (incomplete) phrase,
        re-setting all of its words as unmet.

        :param sentence: A list of generated word  so far
        :return: A deep copy of the object, advanced on word_id.
        """
        
        a_map = predict_dtree.get_tree(nlp,sentence) ## generated text
        met_phrases = []
        for i in range(len(self.root)):
            item=self.root.get_item(i) ## constraints map
            word = item['word']
            roles = item['role'] ## expected dependency role
            if word.lower().strip() not in a_map: ## check whether the expected word is in generated text
                continue
            arr = a_map[word.lower().strip()][0]
            if  arr['dep'] in roles:
                met_phrases.append(item)
        if not met_phrases:
            return self    
        return PositiveState(self.root, met_phrases if met_phrases else None)    
     
class Clause:
    """
    Object used to hold clause.

    :param idx: The id of this clause.
    :param positive: The positive constraints in this clause.
    :param negative: The soft negative constraints in this clause.
    :param satisfy: whether this clause is satisfied
    """

    __slots__ = ('idx', 'positive', 'negative', 'satisfy')

    def __init__(self,
                 idx: int,
                 positive: List[Phrase],
                 negative: List[Phrase],
                 satisfy: float):
        self.idx = idx
        self.positive = positive
        self.negative = negative
        self.satisfy = satisfy

    def __str__(self):
        return f'clause(id={self.idx}, positive={self.positive}, negative={self.negative}, satisfy={self.satisfy})'


def is_prefix(pref: List[int],
              phrase: List[int]):
    if not pref:
        return False
    return pref == phrase[:len(pref)]




class ConstrainedHypothesis:
    """
    Keep track of positive and negative constraint

    hard negative constraint will not be generated in any cases
    soft negative constraint might be generated in some case due to OR gate in clause
    positive constraints will be encourage to appear

    :param constraint_list: A list of clause constraints (each represented as a list of literals).
    """
    def __init__(self,
                 constraint_list: ClauseConstraintList,
                 eos_id: Union[int, list]
                 ):
        self.eos_id = eos_id if isinstance(eos_id, list) else [eos_id]
        self.clauses = []  # type: List[Clause]

        hard_neg_pool, soft_neg_pool, pos_pool = [], [], []  # type: RawConstraintList
        for idx, clause in enumerate(constraint_list):
            if not clause:
                continue
            pos_phrases, neg_phrases = [l[0] for l in clause if l[1]], [l[0] for l in clause if not l[1]]
            # clause contains single negative literal
            if not pos_phrases and len(neg_phrases) == 1:
                hard_neg_pool.extend(neg_phrases)
                #self.clauses.append(Clause(idx=idx, positive=[], negative=neg_phrases, satisfy=True))
            # clause contains multiple negative literals or both negative and positive literals
            elif neg_phrases:
                soft_neg_pool.extend(neg_phrases)
                self.clauses.append(Clause(idx=idx, positive=pos_phrases, negative=neg_phrases, satisfy=True))
            # clause contains only positive literals
            elif pos_phrases and not neg_phrases:
                pos_pool.extend(pos_phrases)
                self.clauses.append(Clause(idx=idx, positive=pos_phrases, negative=[], satisfy=False))
            else:
                import ipdb
                ipdb.set_trace()
                raise ValueError(f'Invalid state {clause}, should not be reached')
        # print('self.clauses',self.clauses)
        # print('pos_pool',pos_pool)
        
        self.hard_negative_state = NegativeState(Trie(hard_neg_pool)) if hard_neg_pool else None
        self.soft_negative_state = NegativeState(Trie(soft_neg_pool)) if soft_neg_pool else None
        # self.positive_state = PositiveState(Trie(pos_pool)) if pos_pool else None
        self.positive_state = PositiveState(Pair(pos_pool)) if pos_pool else None

        # print('positive state:',self.positive_state )

        self.orders = []
        self.in_process = None
        self.max_process = 0

    def __len__(self):
        """
        :return: The number of constraints.
        """
        return len(self.clauses)

    def __str__(self):
        return '\n'.join([str(c) for c in self.clauses])

    def size(self):
        """
        :return: the number of constraints
        """
        return len(self.clauses)

    def num_met(self):
        """
        :return: the number of constraints that have been met.
        """
        if not self.clauses:
            return 0
        return sum([int(c.satisfy) for c in self.clauses])

    def met_order(self):
        """
        :return: the number of constraints that have been met.
        """
        return tuple(sorted(self.orders))

    def clause_in_process(self):
        """
        :return: the index of clause that's in generation.
        """
        return tuple(self.in_process)

    def num_needed(self):
        """
        :return: the number of un-met constraints.
        """
        return self.size() - self.num_met()

    def finished(self):
        """
        Return true if all the constraints have been met.

        :return: True if all the constraints are met.
        """
        return self.num_needed() == 0

    def is_valid(self, wordid: int):
        """
        Ensures </s> is only generated when the hypothesis is completed.

        :param wordid: The wordid to validate.
        :return: True if all constraints are already met or the word ID is not the EOS id.
        """
        return self.finished() or wordid not in self.eos_id

    def avoid(self):
        banned = self.hard_negative_state.avoid() if self.hard_negative_state is not None else set()
        return banned

    def eos(self):
        """
        :return: Return EOS id.
        """
        return self.eos_id

    def advance(self, sentence: str):
        """
        Updates the constraints object based on advancing on word_id.
        If one of literals in a clause is satisfied, we mark this clause as satisfied

        :param word_id: The word ID to advance on.
        """
        obj = copy.deepcopy(self)

        if obj.soft_negative_state is not None:
            raise NotImplementedError

        if obj.hard_negative_state is not None:
            obj.hard_negative_state = obj.hard_negative_state.consume(sentence[-1])
        
        ## this is the current positive constraint in the clasuse
        if obj.positive_state is not None: 
            # print('obj',obj)
            # print('obj.positive_state is not none',obj.positive_state)
            ## this checks the next unsatisfied constraint
            temp_pos_state = obj.positive_state.advance(sentence)
            # print('temp_pos_state here',temp_pos_state)
            ## enters this condition if the constraint is satisfied
            # print('temp_pos_state.met_phrases',temp_pos_state.met_phrases)
            if temp_pos_state.met_phrases is not None:
                # get newly satisfied positive literals
                # print('temp_pos_state.met_phrases',temp_pos_state.met_phrases)
                # phrases_to_delete = []
                newly_met_clause = set()
                for phrase in temp_pos_state.met_phrases:
                    # print('phrase',phrase)
                    for clause in obj.clauses:
                        # print('clasuse',clause.positive)
                        if not clause.satisfy and phrase['word'].strip().lower() in clause.positive[0]:
                            # phrases_to_delete.extend(clause.positive)
                            clause.satisfy = True
                            assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                            newly_met_clause.add(clause.idx)
                obj.orders.extend(sorted(newly_met_clause))
                # print('phrases_to_delete',phrases_to_delete)
                # # delete newly satisfied literals from positive trie state
                # print('temp_pos_state.root',temp_pos_state.root)
                # new_root = copy.deepcopy(temp_pos_state.root)
                # phrases_to_delete = [list(i) for i in set(map(tuple, phrases_to_delete))]
                # for phrase in phrases_to_delete:
                #     if new_root.check_phrase(phrase):
                #         new_root.delete_phrase(phrase)
                # print('new_root',new_root)
                # new_trie_states = set()
                # for state in temp_pos_state.state:
                #     # pointer at root state
                #     if state.parent_trie is None:
                #         new_trie_states.add(new_root)
                #     else:
                #         trace = state.trace_arcs()
                #         new_state = new_root.descend(trace)
                #         if new_state is not None:
                #             new_trie_states.add(new_state)
                # obj.positive_state = PositiveState(positive_trie=new_root)#, state=list(new_trie_states))
                obj.positive_state = temp_pos_state #PositiveState(positive_pair=new_root)#, state=list(new_trie_states))

            else:
                # print('obj.positive_state',obj.positive_state)
                obj.positive_state = temp_pos_state
            # exit()
            # history = [s.trace_arcs() for s in obj.positive_state.state]
            # print('obj.positive_state.root',len(obj.positive_state.root))
            history = []
            for i in range(len(obj.positive_state.root)):
                history.append(obj.positive_state.root.get_item(i))
            # history = [item for item in obj.positive_state.root]
            # print('history',history)
            newly_in_process = set()
            max_process = 0
            for phrase in history:
                for clause in obj.clauses:
                    phrase_in_process = [c for c in clause.positive if is_prefix(phrase, c)]
                    if not clause.satisfy and bool(phrase_in_process):
                        process_portion = len(phrase) / min([len(x) for x in phrase_in_process])
                        max_process = max(max_process, process_portion)
                        assert clause.idx not in obj.orders, 'clause has already satisfied, impossible state'
                        newly_in_process.add(clause.idx)
            obj.in_process = sorted(newly_in_process)
            obj.max_process = max_process
        return obj

    
    

def init_batch(raw_constraints: List[ClauseConstraintList],
               beam_size: int,
               eos_id: Union[int, list]):
    """
    :param raw_constraints: The list of clause constraints.
    :param beam_size: The beam size.
    :param eos_id: The target-language vocabulary ID of the EOS symbol.
    :return: A list of ConstrainedHypothesis objects (shape: (batch_size * beam_size,)).
    """
    constraints_list = [None] * (len(raw_constraints) * beam_size)  # type: List[Optional[ConstrainedHypothesis]]
    for i, raw_list in enumerate(raw_constraints):
        # hyp = ConstrainedHypothesis(raw_list, eos_id)
        # hyp = ConstrainedDtreeHypothesis(raw_list, eos_id)
        hyp = ConstrainedHypothesis(raw_list, eos_id)
        # print('hyp is>>>>',hyp,'<<<<')
        

        idx = i * beam_size
        constraints_list[idx:idx + beam_size] = [copy.deepcopy(hyp) for _ in range(beam_size)]
    return constraints_list

## CHECK This Moha
class ConstrainedCandidate:
    """
    Object used to hold candidates for the beam in topk().

    :param row: The row in the scores matrix.
    :param col: The column (word ID) in the scores matrix.
    :param score: the associated accumulated score.
    :param hypothesis: The ConstrainedHypothesis containing information about met constraints.
    """

    __slots__ = ('row', 'col', 'score', 'hypothesis', 'rank')

    def __init__(self,
                 row: int,
                 col: int,
                 score: float,
                 hypothesis: ConstrainedHypothesis,
                 rank: float = None,):
        self.row = row
        self.col = col
        self.score = score
        self.hypothesis = hypothesis
        self.rank = rank

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __str__(self):
        return '({}, {}, {}, {})'.format(self.row, self.col, self.score, self.hypothesis.num_met())


if __name__ == '__main__':
    # clauses = [[[([3, 4, 5], True), ([3, 4], True), ([4, 6], True)], [([3, 4], True), ([6], True), ([7], True)]],
    #            [[([6], True), ([6, 7], True), ([6, 7, 8], True)], [([6, 9], True), ([6, 4, 9], True)]],
    #            [[([3, 4, 5], True)], [([3, 4], True)], [([4, 5], True)]],
    #            [[([3, 4], True)], [([2, 3, 5], True)], [([6, 5], True)]]]
    
    # clauses = [[[(['cat','catches','ball'], True)]]]
    # clauses=[[[(["team","nsubj"],True),(["team","csubj:pass"],True),(["team","cc:preconj"],True)],[(["run","root"],True),(["run","csubj"],True),(["run","ccomp"],True),(["run","xcomp"],True),(["run","acl"],True)],[(["field","nsubj:pass"],True),(["field","obj"],True),(["field","obl"],True),(["field","iobj"],True),(["field","cc:preconj"],True)]]]
    clauses=[[[(["team","csubj:pass","nsubj"],True)],[(["run","root","csubj"],True)],[(["field","obj","nsubj:pass"],True)]]]

    print('init batch')
    constraints = init_batch(raw_constraints=clauses,
                             beam_size=1,
                             eos_id=0)

    # constraint = constraints[2]
    constraint = constraints[0]

    print('constraints is:',constraint)
    print(constraints)
    print()
    # exit()
    words=[2]
    # strr='the cat catches the large round green ball .'
    strr = 'The team will run across the field.'
    words=strr.split(' ')
    for i in range(1,len(words)):
        print('------------%s----------'%str(words[:i]))
        # constraint = constraint.advance_dtree(' '.join(words[:i]))
        constraint = constraint.advance(' '.join(words[:i]))
        print('constraint>>>\n',constraint)
        # print('constraint.positive_state>>>\n',constraint.positive_dtree.met)



