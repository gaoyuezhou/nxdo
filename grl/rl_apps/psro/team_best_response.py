import pyspiel
import numpy as np
import gurobipy as grb
from gurobipy import GRB


class TeamBestResponder(object):
    """This objects constructs a best responder for a given `team`"""

    def __init__(self, game, team):
        assert team in [0, 1]

        self.team = team
        self.game_ = game

        env = grb.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        self.model_ = grb.Model(env=env)

        # List of (chance*team_payoff, [pl0_seq, pl1_seq, pl2_seq, pl3_seq]) triplets.
        # Each sequence is the terminal (infostate, action) for that team
        self.leaves_ = []

        # For each sequence of the `self.team`, we introduce a variable that contains
        # the product of the actions for the team on the path to that leaf. This
        # corresponds to a *sequence-form* strategy
        self.team_reaches_ = {}

        # For each leaf in order, we create a GRB variable that corresponds to the
        # reach of that leaf from the self.team's point of view
        self.leaf_reaches_ = []

        # For each player, we store the the treeplex, in the form of sequence -> {infostate: [children]}
        self.tpxs_ = [{}, {}, {}, {}]

        self.preprocess_game_()

    def preprocess_game_(self):
        # First, we populate all of self.leaves_
        print("[TeamBestResponder] Constructing treeplexes...")
        self.traverse_(self.game_.new_initial_state(),
                       1.0,
                       [(0, None, None), (1, None, None), (2, None, None), (3, None, None)])

        print("[TeamBestResponder] ... done")
        # Allocate variables for each of self.team's sequences
        team_players = [p for p in range(4) if p % 2 == self.team]
        for player in team_players:
            self.team_reaches_[(player, None, None)] = 1.0  # Empty sequence
            for (parent, infostate_children) in self.tpxs_[player].items():
                assert parent in self.team_reaches_

                for infostate_, children in infostate_children.items():
                    s = 0
                    for child in children:
                        assert child[0] == player
                        assert child[1] == infostate_

                        assert child not in self.team_reaches_

                        self.team_reaches_[child] = self.model_.addVar(
                            0, 1, vtype=GRB.BINARY)
                        s += self.team_reaches_[child]
                    # Sequence-form constraint
                    self.model_.addConstr(s == self.team_reaches_[parent])

        for (_, pl_seqs) in self.leaves_:
            leaf_reach = self.model_.addVar(0, 1, vtype=GRB.BINARY)
            self.leaf_reaches_.append(leaf_reach)

            s = 0
            for player in team_players:
                assert pl_seqs[player] in self.team_reaches_

                pl_reach = self.team_reaches_[pl_seqs[player]]
                self.model_.addConstr(leaf_reach <= pl_reach)
                s += pl_reach
            self.model_.addConstr(leaf_reach >= s - 1)

    def best_response(self, opposing_team_policy_mixture):
        """Computes the best response for `self.team`.

        Returns a pair (br_value, br_policy).
        """

        opposing_team_players = [p for p in range(4) if p % 2 != self.team]
        assert all([set(policy.keys()) == set(opposing_team_players)
                   for _, policy in opposing_team_policy_mixture])

        # For each sequence of opposing team, compute the reaches
        opposing_team_reaches = {}
        mix_sum = 0.0
        for (mix_prob, policy) in opposing_team_policy_mixture:
            mix_sum += mix_prob

            for player in opposing_team_players:
                opposing_team_reaches[(player, None, None)] = 1.0
                for (parent, infostate_children) in self.tpxs_[player].items():
                    assert parent in opposing_team_reaches
                    parent_reach = opposing_team_reaches[parent]

                    for infostate, children in infostate_children.items():
                        for child in children:
                            (player_, infostate_, action) = child
                            assert player_ == player
                            assert infostate_ == infostate

                            if child not in opposing_team_reaches:
                                opposing_team_reaches[child] = 0.0

                            opposing_team_reaches[child] += parent_reach * \
                                mix_prob * policy[player][infostate][action]

        assert abs(mix_sum - 1.0) < 1e-9

        obj = 0
        for reach_var, (payoff, pl_seqs) in zip(self.leaf_reaches_, self.leaves_):
            opposing_team_reach = 1.0
            for player in opposing_team_players:
                opposing_team_reach *= opposing_team_reaches[pl_seqs[player]]

            obj += payoff * opposing_team_reach * reach_var
        self.model_.setObjective(obj, sense=GRB.MAXIMIZE)

        self.model_.optimize()
        assert self.model_.status == GRB.OPTIMAL

        obj_val = self.model_.getAttr(GRB.Attr.ObjVal)
        team_policy = {p: self.get_policy_(p)
                       for p in range(4) if p % 2 == self.team}
        return (obj_val, team_policy)

    def get_policy_(self, player):
        out = {}
        for infostate_children in self.tpxs_[player].values():
            for infostate, children in infostate_children.items():
                assert infostate not in out
                out[infostate] = {}

                last_action = None
                for child in children:
                    assert child[0] == player
                    assert child[1] == infostate
                    action = child[2]
                    out[infostate][action] = self.team_reaches_[
                        child].getAttr(GRB.Attr.X)
                    last_action = action
                if sum(out[infostate].values()) < 1e-3:
                    # Convert back to behavioral policy by picking the last action
                    out[infostate][last_action] = 1.0
        return out

    def traverse_(self, state, chance_reach, pl_seqs):
        if state.is_terminal():
            returns = state.returns()
            team_payoff = sum([returns[player]
                              for player in [0, 1, 2, 3] if player % 2 == self.team])
            self.leaves_.append((chance_reach * team_payoff, pl_seqs))
        elif state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            for (action, prob) in outcomes_with_probs:
                self.traverse_(state.child(action), chance_reach*prob, pl_seqs)
        else:
            # Player node
            player = state.current_player()
            pl_seq = pl_seqs[player]
            infostate = state.information_state_string()

            state_seqs = [(player, infostate, action)
                          for action in state.legal_actions()]

            if pl_seq not in self.tpxs_[player]:
                self.tpxs_[player][pl_seq] = {infostate: state_seqs}
            elif infostate not in self.tpxs_[player][pl_seq]:
                self.tpxs_[player][pl_seq][infostate] = state_seqs
            else:
                assert self.tpxs_[player][pl_seq][infostate] == state_seqs

            for action in state.legal_actions():
                new_seqs = pl_seqs[:]
                this_seq = (player, infostate, action)
                new_seqs[player] = this_seq

                self.traverse_(state.child(action), chance_reach, new_seqs)


def expected_value(game, policies):
    assert set(policies.keys()) == set([0, 1, 2, 3])

    def traverse_(state, reach):
        ans = np.zeros(2)

        if state.is_terminal():
            returns = state.returns()
            t0_payoff = returns[0] + returns[2]
            t1_payoff = returns[1] + returns[3]

            ans[0] = t0_payoff * reach
            ans[1] = t1_payoff * reach
        elif state.is_chance_node():
            outcomes_with_probs = state.chance_outcomes()
            for (action, prob) in outcomes_with_probs:
                ans += traverse_(state.child(action), reach*prob)
        else:
            # Player node
            player = state.current_player()
            infostate = state.information_state_string()
            for action in state.legal_actions():
                ans += traverse_(state.child(action), reach *
                                 policies[player][infostate][action])
        return ans
    return traverse_(game.new_initial_state(), 1.0)


def uniform_policy(game, player):
    policy = {}

    def traverse_(state):
        actions = state.legal_actions()

        if state.current_player() == player:
            policy[state.information_state_string()] = {
                action: 1.0 / len(actions) for action in actions
            }

        for action in state.legal_actions():
            traverse_(state.child(action))
    traverse_(game.new_initial_state())
    return policy


if __name__ == '__main__':
    #     game = pyspiel.load_game(
    #         "goofspiel(players=4,num_cards=4,points_order=ascending,imp_info=True)")
    #     game = pyspiel.convert_to_turn_based(game)

    #     root = game.new_initial_state()
    #     s = root.child(0)
    #     s = s.child(0)
    #     s = s.child(0)
    #     s = s.child(1)
    #     assert s.current_player() == 0
    #     print(s.information_state_string())
    #     s = s.child(1)
    #     s = s.child(2)
    #     s = s.child(3)
    #     s = s.child(2)
    #     print(s.information_state_string())

    #     s = root.child(0)
    #     s = s.child(0)
    #     s = s.child(0)
    #     s = s.child(2)
    #     assert s.current_player() == 0
    #     print(s.information_state_string())
    #     s = s.child(1)
    #     s = s.child(2)
    #     s = s.child(3)
    #     s = s.child(1)
    #     print(s.information_state_string())

    # else:
    game = pyspiel.load_game(
        "kuhn_poker(players=4)")
    # if game.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    #     game = pyspiel.convert_to_turn_based(game)
    # game = pyspiel.load_game(
    #     "goofspiel(players=4,num_cards=3,points_order=ascending,imp_info=True)")
    # game = pyspiel.convert_to_turn_based(game)
    br = TeamBestResponder(game, 0)

    policies = {p: uniform_policy(game, p) for p in [1, 3]}
    opposing_team_mixture = [(1.0, policies)]

    print(br.best_response(opposing_team_mixture))
    policies = {p: uniform_policy(game, p) for p in [0, 1, 2, 3]}
    print(expected_value(game, policies))