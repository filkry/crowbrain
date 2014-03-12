import rate_model, rate_bernoulli_decay_model, rate_bernoulli_decay_participants
import rate_bernoulli_comparison, siam_time, split_originality_blend_model

def print_model(f, model_string, title, label):
    print('\\section{%s}' % title, file=f)
    print('\\label{sec:%s}' % label, file=f)
    print('\\begin{lstlisting}[numbers=left,frame=single,breaklines=true,basicstyle=\small]',
            file=f)
    print(model_string, file=f)
    print('\\end{lstlisting}', file=f)
    print('\n\n', file=f)

if __name__ == '__main__':
    with open('tex/model_appendix.tex', 'w') as f:
        print('\\chapter{Stan specification for models}\n\n', file=f)

        print_model(f, rate_model.model_string, 'Exponential decay model',
                'exponential_decay')
        print_model(f, rate_bernoulli_decay_model.model_string,
                'Decaying Bernoulli model', 'decaying_bernoulli')
        print_model(f, rate_bernoulli_decay_participants.model_string,
                'Decaying Bernoulli model with participant parameters',
                'decaying_bernoulli_part')
        print_model(f, rate_bernoulli_comparison.model_string,
                'Comparison model of exponential decay and decaying Bernoulli',
                'exp_bern_comp')
        print_model(f, split_originality_blend_model.model_string,
                'Novelty within brainstorming run model',
                'novelty_run_model')
        print_model(f, siam_time.model_string,
                'Idea generation time model',
                'idea_time_model')
