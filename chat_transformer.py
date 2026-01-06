from transformer_model_llama_june2025 import TransformerModel
import torch
import torch.nn.functional as F

from tokenizers import ByteLevelBPETokenizer



model_name = "model_finetuned_no_reason.pt"

question_end_token = 1
answer_end_token = 2
think_start_token = 7
think_end_token = 8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model, tokenizer, tokens, max_length=100, temperature=0.7, top_p=0.9,
                  repetition_penalty=1.1, stream=False):
    model.clear_kv_cache()
    

    # print("prompt->", prompt)

    generated = tokens
    # for g in range(generated.shape[1]):
    #     model(generated[:, g:g+1], start_pos=g)
    
    last_token = 0
    last_recent_tokens = []
    new_generated = []
    with torch.no_grad():
        # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(generated[:, 0:-1], start_pos=0)


        for tokens_generated in range(max_length):
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(generated[:, -1:], start_pos=generated.shape[1])

            next_token_logits = outputs[0, :] / temperature

            # Apply repetition penalty
            rp = repetition_penalty

            if len(last_recent_tokens) > len(set(last_recent_tokens))*3:
                rp*=3
                print("extra penalty")
            for token_id in set(last_recent_tokens):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= rp
                else:
                    next_token_logits[0, token_id] *= rp

            # if tokens_generated < 3:
            #     next_token_logits[0, answer_end_token] -= 100
            # if tokens_generated == 0:
            #     next_token_logits[0, generated[0, 0]] -= 100


            # newline_tokens = [208, 230, 15078, 19]#, 35, 23, 1820]
            # if generated[:, -1].item() in newline_tokens and generated[:, -2].item() in newline_tokens: # bug where it generates new line forever
            #     for n in newline_tokens:
            #         next_token_logits[0, n] -= 1000
                # print("PREVENTING TOKEN")

            # print("Gen->",generated[:, -1].item(), generated[:, -2].item())
           

            next_token_logits = next_token_logits.squeeze()
            filtered_logits = top_p_filtering(next_token_logits, top_p=top_p)
            probabilities = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probabilities, 1)


            # if next_token != last_token:
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            last_recent_tokens.append(next_token.item())


            if len(last_recent_tokens)>200:
                last_recent_tokens = last_recent_tokens[1::]
            # print("next", next_token)
            last_token = next_token
            if last_token == answer_end_token:
                break

            new_generated.append(next_token.item())

            if stream:
                print(tokenizer.decode([next_token.item()]), end='', flush=True)


    return tokenizer.decode(new_generated)


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits



def chat_template(conversation):
    tokens = []

    for conv in conversation:

        if conv["message"][0] != "\n":
            conv["message"] = "\n" + conv["message"]
        # print(conv)
        tokens += tokenizer.encode(conv["message"]).ids

        if conv["role"] == "user":
            tokens += [question_end_token, 208]
            # print("user", tokens)
        elif conv["role"] == "assistant":
            tokens += [answer_end_token]



    return tokens




tokenizer = ByteLevelBPETokenizer.from_file(
    vocab_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-vocab.json",
    merges_filename="my_tokenizer_50k_2025/tokenizer_50k_2025-merges.txt"
)
# print(tokenizer.encode('Hello\n\n\n\n').ids)




# print(tokenizer.decode([0,1,2,3,4,5,6,7,8]))


nlayers = 24  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 16  # number of heads in ``nn.MultiheadAttention``
# ntokens = 20000
dim = 1024+512
ffn_dim = dim*4
rope_theta=50000
ntokens=50000
max_seq_len=2048
batch_size=1

# model = TransformerModel(ntokens, max_seq_len, emsize, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=batch_size, device=device).to(device)
model = TransformerModel(ntokens, max_seq_len, -1, nhead, nlayers, ffn_dim=ffn_dim, dim=dim, batch_size=batch_size, device=torch.device("cpu"))
model.eval()



model.load_state_dict(torch.load(model_name, weights_only=True, map_location=torch.device("cpu")))
model.dtype = torch.bfloat16
if torch.cuda.is_available():
    model.device = device
    model = model.to(device)


conversation = []
full_text = ""
while True:
    torch.manual_seed(0)

    prompt = input("\n>>")
    # prompt = "What is faster, a turtle or a cheetah?"

    if len(prompt) == 0:
       
        model.clear_kv_cache()
        print("reset")
        full_text = ""
        conversation = []
        continue



    conversation.append({"role": "user", "message": prompt})

    # print(chat_template(conversation), tokenizer.decode(chat_template(conversation)))
    tokens = torch.tensor([chat_template(conversation)], dtype=torch.long).to(device)

    response = generate_text(model, tokenizer, tokens, stream=True, temperature=0.6, top_p=0.9, max_length=2048, repetition_penalty=1.2) 


    conversation.append({"role": "assistant", "message": response})

