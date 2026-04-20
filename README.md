# RIFT

```shell
conda activate add-thin
```

### Current issue
1. loss change is too small. (Have already fixed. By changing the learning rate and move omega MLP)

2. test sample will generate the large omega. (set the limitation of omega value, but cause another issue -> every omega value is same, so the samples are same, the was distances are same.)

3. L11 Loss is unstable, up to infinity. Reason is that the forward sample scalar will generate the extreme value, cause this happen. (still working)

### 4/01/2026 Update
1. Erase the omega MLP in the DiT process, and create this layer in the init of AddThin class

    ```python 
        x_feat = x.squeeze(1)          # [1, 9]
        raw = self.omega_head(x_feat)  # [1, 1]
        return raw.squeeze(-1).mean()   
    ```

2. Change the loss weigth. From the ``final_loss = alpha * mse_loss - beta * l11_loss`` to 

    ```python
        alpha = 1.0
        beta = 0.1
        final_loss = alpha * mse_loss - beta * l11_loss
    ```

3. Change the learning Rate. ``learning_rate: 0.001``

4. due to the omega is too large, then create a constrain that limit the value of omega scalar.
    ```python 
        raw = self.DiT(batch_e, time_emb[batch_index].to(device), dt)
        omega_k_minus_1 = 1.0 + torch.sigmoid(raw)
    ```
5. remove the final layer in DiT.


### 3/30/2026 Update

1. Set the chunk size to sample

    ```python 
        chunk_size = 512
        for b in range(B):
            Nb = int(n_add[b].item())
            accepted_chunks = []

            if Nb > 0:
                for start in range(0, Nb, chunk_size):
                    curr_size = min(chunk_size, Nb - start)
                    e_chunk = torch.rand(curr_size, device=device) * T   # [curr_size]

                    diff = e_chunk[:, None] - e_0[b][None, :]            # [curr_size, L]
                    kernel = torch.exp(-diff**2 / bandwidth_square)
                    lambda0_add = kernel.mean(dim=1)                     # [curr_size]

                    omega_add = (lambda_1 ** (r_k - r_k1)) * (lambda0_add ** (r_k1 - r_k))
                    lambda_bar = (lambda_1 ** r_k1) * (lambda0_add ** (1 - r_k1))

                    p = (omega_add - 1) * lambda_bar / upper[b]
                    p = torch.clamp(p, 0, 1)

                    accept = torch.rand_like(p) < p
                    accepted_chunks.append(e_chunk[accept])

                e_add = (
                    torch.cat(accepted_chunks)
                    if len(accepted_chunks) > 0
                    else torch.empty(0, device=device)
                )
            else:
                e_add = torch.empty(0, device=device)
    ```

