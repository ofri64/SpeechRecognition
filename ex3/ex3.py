import sys
import parse_data
import ctc_forward

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise TypeError("The program takes exactly 3 arguments - path to matrix (.npy file), "
                        "a string representing the labeling to we wish to calculate the probability for,"
                        "and a string representing our alphabet")
    matrix_file = sys.argv[1]
    transcript = sys.argv[2]
    alphabet_string = sys.argv[3]

    output_mat = parse_data.load_matrix(matrix_file)
    modified_transcript = parse_data.get_modified_transcript(transcript)
    alpha_to_index = parse_data.get_alphabet_index_translation(alphabet_string)

    prob = ctc_forward.get_transcript_prob(output_mat, modified_transcript, alpha_to_index)
    prob_two_digits_with_round = "{0:.2f}".format(prob)
    print(prob_two_digits_with_round)
