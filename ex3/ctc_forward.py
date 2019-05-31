from parse_data import EMPTY


def get_transcript_prob(outputs_mat, transcript, alphabet_index_trans):
    alpha = {}

    # the notation (i,j) stands for (s=i, t=j)

    # base cases
    alpha[(0, 0)] = outputs_mat[0][alphabet_index_trans[transcript[0]]]  # transcript[0] == EMPTY token
    alpha[(1, 0)] = outputs_mat[0][alphabet_index_trans[transcript[1]]]
    for s in range(2, len(transcript)):
        alpha[(s, 0)] = 0  # can start only from EMPTY or first real token in transcript

    # dynamic programming computation
    T = len(outputs_mat)
    l = len(transcript)
    for t in range(1, T):
        for s in range(l):
            alpha[(s, t)] = compute_alpha_s_t(s, t, alpha, outputs_mat, transcript, alphabet_index_trans)

    print(alpha)
    return alpha[(l - 1, T - 1)] + alpha[(l - 2, T - 1)]


def compute_alpha_s_t(s, t, alpha_dict, outputs_mat, transcript, alphabet_index_trans):
    # base case
    if s < 0:
        return 0

    modified_transcript_len = len(transcript)
    original_transcript_len = len(outputs_mat)  # also known as T
    if s < modified_transcript_len - 2 * (original_transcript_len - t):
        return 0

    # recursion
    t_s = transcript[s]
    if t_s == EMPTY:
        return (alpha_s_t(s, t - 1, alpha_dict) + alpha_s_t(s - 1, t - 1, alpha_dict)) * outputs_mat[
            t, alphabet_index_trans[t_s]]

    if s > 1 and transcript[s] == transcript[s - 2]:
        return (alpha_s_t(s, t - 1, alpha_dict) + alpha_s_t(s - 1, t - 1, alpha_dict)) * outputs_mat[
            t, alphabet_index_trans[t_s]]

    else:
        return (alpha_s_t(s, t - 1, alpha_dict) + alpha_s_t(s - 1, t - 1, alpha_dict) + alpha_s_t(s - 2, t - 1, alpha_dict)) * outputs_mat[
                   t, alphabet_index_trans[t_s]]


def alpha_s_t(s, t, alpha_dict):
    if s < 0:
        return 0

    return alpha_dict[(s, t)]
