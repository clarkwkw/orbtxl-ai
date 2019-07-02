from ..structs import GameAction
import tensorflow as tf
import numpy as np

SAVE_FILE_NAME = "savefile.ckpt"
PARAMETERS_FILE_NAME = "paras.json"


class PolicyGradientModel:
    def __init__(
        self,
        actions,
        sample_shape,
        from_save=None,
        learning_rate=0.01,
        dropout=0.2,
        reward_decay=0.95
    ):
        self.__graph = tf.Graph()
        self.__sess = tf.Session(graph=self.__graph)
        self.__actions = actions
        with self.__graph.as_default() as g:
            if from_save is None:
                self.__build_graph(g, actions, sample_shape, dropout, learning_rate)
                self.__reward_decay = reward_decay
                self.__sess.run(tf.initializers.global_variables())
            else:
                saver = tf.train.import_meta_graph(
                    from_save.rstrip("/") + "/" + SAVE_FILE_NAME + ".meta"
                )
                saver.restore(
                    self.__sess, from_save.rstrip("/") + "/" + SAVE_FILE_NAME
                )
                self.__obs = g.get_tensor_by_name("inputs/observations:0")
                self.__acts = g.get_tensor_by_name("inputs/actions_num:0")
                self.__vt = g.get_tensor_by_name("inputs/actions_value:0")
                self.__a_filter = g.get_tensor_by_name(
                    "inputs/actions_filter:0"
                )
                self.__all_act_prob = tf.get_collection("all_act_prob")[0]
                self.__loss = tf.get_collection("loss")[0]
                self.__train_op = tf.get_collection("train_op")[0]

    def __build_graph(self, graph, actions, sample_shape, dropout, learning_rate):

        with tf.name_scope('inputs'):
            self.__obs = tf.placeholder(
                tf.float32, [None] + sample_shape + [4], name="observations"
            )
            self.__acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num"
            )
            self.__vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value"
            )
            self.__a_filter = tf.placeholder(
                tf.float32, [None, len(actions)], name="actions_filter"
            )

        conv1 = tf.layers.conv2d(self.__obs, 10, 30, activation=tf.nn.relu)
        conv1 = tf.compat.v1.layers.max_pooling2d(conv1, 5, 2)

        conv2 = tf.layers.conv2d(conv1, 20, 10, activation=tf.nn.relu)
        conv2 = tf.compat.v1.layers.max_pooling2d(conv2, 5, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.compat.v1.layers.dense(fc1, 1024)
        fc1 = tf.compat.v1.layers.dropout(fc1, rate=dropout, training=True)
        result = tf.compat.v1.layers.dense(fc1, len(actions))
        self.__all_act_prob = tf.nn.softmax(result)

        with tf.name_scope('loss'):
            '''
                to maximize total reward (log_p * R)
                is to minimize -(log_p * R),
                and the tf only have minimize(loss)
            '''
            # this is negative log of chosen action
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=result,
                labels=self.__acts
            )
            # reward guided loss
            self.__loss = tf.reduce_mean(neg_log_prob * self.__vt)

        with tf.name_scope('train'):
            self.__train_op = tf.compat.v1.train.AdamOptimizer(learning_rate)\
                                .minimize(self.__loss)

        graph.add_to_collection("all_act_prob", self.__all_act_prob)
        graph.add_to_collection("loss", self.__loss)
        graph.add_to_collection("train_op", self.__train_op)

    def get_action(self, screenshot):
        prob_weights = self.__sess.run(
            self.__all_act_prob,
            feed_dict={
                self.__obs: np.array(screenshot)[np.newaxis, :, :]
            }
        )
        print(prob_weights)
        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel()
        )  # select action w.r.t the actions prob
        print("action:", action)
        return GameAction(self.__actions[action])

    def on_game_ended(self, game_record):
        def discount_and_norm_rewards():
            # discount episode rewards
            episode_rewards = [0 for _ in game_record.time_points]
            episode_rewards[-1] = game_record.score
            discounted_ep_rs = np.zeros_like(
                episode_rewards,
                dtype=np.float32
            )

            running_add = 0
            for t in reversed(range(0, len(episode_rewards))):
                running_add = running_add * self.__reward_decay \
                            + episode_rewards[t]
                discounted_ep_rs[t] = running_add

            # normalize episode rewards
            discounted_ep_rs -= np.mean(discounted_ep_rs)
            std = np.std(discounted_ep_rs)
            if std >= 1e-3:
                discounted_ep_rs /= std
            return discounted_ep_rs

        discounted_ep_rs_norm = discount_and_norm_rewards()

        observations = [
            np.array(tp.screenshot) for tp in game_record.time_points
        ]
        actions = [
            self.__actions.index(tp.action.press_time)
            for tp in game_record.time_points
        ]

        _, loss = self.__sess.run(
            [self.__train_op, self.__loss],
            feed_dict={
                self.__obs: np.stack(observations, axis=0),
                self.__acts: np.asarray(actions),
                self.__vt: discounted_ep_rs_norm
            }
        )
