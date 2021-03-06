### Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing

**Paper name:** Closed-Form Factorization of Latent Semantics in GANs
**Publication:** CVPR2021
**Problem:** image local editing and image interpolation
**motivation:** Due to the absence of spatial dimensions in the latent space, an encoder compresses an image's local semantics into a vector in an entangled manner, making it difficult to reconstruct the image. Instead of learning a vector-based latent representation, they utilize a tensor with explicit spatial dimensions. their proposed representation benefits from its spatial dimensions, enabling GANs to easily encode the local semantics of images into the latent
**Solution:**

Their goal is to project images to a latent space accurately with an encoder in real-time and locally manipulate images on the latent space. They propose StyleMap-GAN which adopts stylemap, an intermediate latent space with spatial dimensions, and a spatially variant modulation based on the stylemap.

**modifications on styleGAN** 

<img src="./image/image-20210531090223809.png" alt="image-20210531090223809" style="zoom:50%;" />

1. a stylemap with spatial dimensions instead of style vectors to control feature maps.
2. modulation operation:$h_{i+1}=\left(\gamma_{i} \otimes \frac{h_{i}-\mu_{i}}{\sigma_{i}}\right) \oplus \beta_{i}$
3. remove per-pixel noise which was an extra source of spatially varying inputs in styleGAN.

**overall train scheme**

<img src="./image/image-20210531091108857.png" alt="image-20210531091108857" style="zoom:50%;" />

1. F, G, E, and D are indicated the mapping network, synthesis network with stylemap resizer, encoder, and discriminator, respectively. 

2. All module are joint learned using following losses:

   <img src="./image/image-20210531091603516.png" alt="image-20210531091603516" style="zoom:50%;" />

3. local editing: $\ddot{\mathbf{w}}=\mathbf{m} \otimes \widetilde{\mathbf{w}} \oplus(1-\mathbf{m}) \otimes \mathbf{w}$

**Experiments**

1. Comparison of reconstruction and generation quality across different resolutions of the stylemap.

   <img src="./image/image-20210531092007393.png" alt="image-20210531092007393" style="zoom:50%;" />

2. Comparison with the baselines for real image projection:

   1. <img src="./image/image-20210531092053954.png" alt="image-20210531092053954" style="zoom:50%;" />

   2. Image2StyleGAN:

      <img src="./image/image-20210531092232453.png" alt="image-20210531092232453" style="zoom:50%;" />

   3. Image2StyleGAN++:

      <img src="./image/image-20210531092327707.png" alt="image-20210531092327707" style="zoom:50%;" />

3. <img src="./image/image-20210531092500281.png" alt="image-20210531092500281" style="zoom:50%;" />

4. <img src="./image/image-20210531092533312.png" alt="image-20210531092533312" style="zoom:50%;" />

5. <img src="./image/image-20210531092627568.png" alt="image-20210531092627568" style="zoom:50%;" />

### FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery

**Paper name:** FineGAN: Unsupervised Hierarchical Disentanglement for Fine-Grained Object Generation and Discovery
**Publication:** CVPR2019
**Problem:** ??????????????????????????????????????? Disentangle factors without supervision to generate realistic and diverse images for fine-grained classes.

**motivation:** ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

**Solution:**

FineGAN????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

**Hierarchical finegrained disentanglement**

![image-20210531094305068](./image/image-20210531094305068.png)

1. ??????: $z \sim \mathcal{N}(0,1)$ $b \sim \operatorname{Cat}\left(K=N_{b}, p=1 / N_{b}\right)$ $p \sim \operatorname{Cat}\left(K=N_{p}, p=1 / N_{p}\right)$ $c \sim \operatorname{Cat}\left(K=N_{c}, p=1 / N_{c}\right)$?????????b,p,c???onehot?????????$N_{b},N_{p},N_{c}$???????????????????????????????????????????????????

2. ???????????????????????????: ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

3. ?????????????????????

   1. ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????$N \times N$???patch?????????patch?????????????????????????????????????????????????????????$D_{b}$?????????patch?????????????????????loss??????????????????????????????????????????????????????

      $\mathcal{L}_{b g_{-} a d v}=\min _{G_{b}} \max _{D_{b}} \mathbb{E}_{x}\left[\log \left(D_{b}(x)\right)\right]+\mathbb{E}_{z, b} \mid\left[\log \left(1-D_{b}\left(G_{b}(z, b)\right)\right)\right]$

      ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????$D_{aux}$????????????????????????????????????$G_b$ ????????????$\mathcal{L}_{b g_{-} a u x}=\min _{G_{b}} \mathbb{E}_{z, b}\left[\log \left(1-D_{a u x}\left(G_{b}(z, b)\right)\right)\right]$

4. ?????????(??????????????????)

   1. ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

   2. ?????????????????????????????????????????????????????????????????????$\mathcal{B}$????????????????????????????????????????????????$G_{p, f}$???$G_{p, m}$??????????????????$\mathcal{P}_{f}$???????????????$\mathcal{P}_{m}$???????????????????????????$\mathcal{P}=\mathcal{P}_{f, m}+\mathcal{B}_{m}$????????? $\mathcal{P}_{f, m}=\mathcal{P}_{m} \odot \mathcal{P}_{f}$ ???$\mathcal{B}_{m}=\left(1-\mathcal{P}_{m}\right) \odot \mathcal{B}$

   3. ??????????????????????????????$D_{p}$ ?????????????????????????????????????????????????????????????????????????????????InfoGAN??????????????????????????????$p$???????????????${P}_{f, m}$??????????????????????????????????????????????????????????????????????????????????????????$D_p$??????????????????$G_{p, f}, G_{p, m}$???????????????????????????????????????????????????????????????**????????????????????????generator????????????????????? ?????????????????????????????????????????????p,(????????????onehot?????????????????????????????????)**

      ???????????????????????????????????????$\mathcal{L}_{p}=\mathcal{L}_{p_{-} i n f o}=\max _{D_{p}, G_{p, f}, G_{p, m}} \mathbb{E}_{z, p}\left[\log D_{p}\left(p \mid \mathcal{P}_{f, m}\right)\right]$
      
      info???????????????????????? criterion_class??????????????????????????????p_code??????????????????pred_p??????discriminator????????????????????????????????????????????????????????????????????????
      
      ```python
      if i == 1: # Mutual information loss for the parent stage (1)
              pred_p = self.netsD[i](self.fg_mk[i-1])
              errG_info = criterion_class(pred_p[0], torch.nonzero(p_code.long())[:,1])
      elif i == 2: # Mutual information loss for the child stage (2)
              pred_c = self.netsD[i](self.fg_mk[i-1])
              errG_info = criterion_class(pred_c[0], torch.nonzero(c_code.long())[:,1]
      ```
      
      

5. ?????????

   1. ????????????????????????????????????????????????????????????????????????????????????$G_c$??????????????????????????????????????????$G_{c, f}$  $G_{c, m}$??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????$\mathcal{C}=\mathcal{C}_{f, m}+\mathcal{P}_{c, m}$ $\mathcal{C}_{f, m}=\mathcal{C}_{m} \odot \mathcal{C}_{f}$, $\mathcal{P}_{c, m}=\left(1-\mathcal{C}_{m}\right) \odot \mathcal{P}$
   2. ???????????????loss????????????????????????????????????????????????:????????????????????????????????????????????????????????????????????????$\mathcal{L}_{c_{-} i n f o}=\max _{D_{c}, G_{c, f}, G_{c, m}} \mathbb{E}_{z, p, c}\left[\log D_{c}\left(c \mid \mathcal{C}_{f, m}\right)\right]$?????????????????????????????????????????????????????????????????????????????????????????????????????????$\mathcal{L}_{c_{-} a d v}=\min _{G_{c}} \max _{D_{a d v}} \mathbb{E}_{x}\left[\log \left(D_{a d v}(x)\right)\right]+\mathbb{E}_{z, b, p, c}\left[\log \left(1-D_{a d v}(\mathcal{C})\right)\right]$

   ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

   **experiments**

   <img src="./image/image-20210603194536162.png" alt="image-20210603194536162" style="zoom:50%;" />

   <img src="./image/image-20210603194556218.png" alt="image-20210603194556218" style="zoom:50%;" />

   

   <img src="./image/image-20210603194610650.png" alt="image-20210603194610650" style="zoom:50%;" />



### GENERATING FURRY CARS: DISENTANGLING OBJECT SHAPE & APPEARANCE ACROSS MULTIPLE DOMAINS

**Paper name:** ENERATING FURRY CARS: DISENTANGLING OBJECT SHAPE & APPEARANCE ACROSS MULTIPLE DOMAINS
**Publication:** iclr2021
**Problem:** gan
**motivation:** ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

**method:**

<img src="./image/image-20210603200106268.png" alt="image-20210603200106268" style="zoom: 50%;" />

<img src="./image/image-20210603195033365.png" alt="image-20210603195033365" style="zoom:50%;" />

1. FineGAN:
   
   1. ??????????????????????????????????????????????????????????????????????????????????????????e??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
   
2. Learning a differentiable histogram of low-level features.

   1. ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

   2. ????????????????????????????????????????????????????????????????????????filter bank?????????????????????????????????????????????mask??????????????????????????????????????????????????????????????????????????????$k$???????????????$h$??????????????????$k$????????????????????????????????????

   3. ?????????????????????$h$???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????$I=G\left(x_{1}, y_{1}, z_{1}, b_{1}\right)$?????????$z$????????????pose????????????$I_{\text {pos }_{1}}=G\left(x_{1}, y_{1}, z_{2}, b_{1}\right)$???????????????????????????????????????????????????,???????????????NT-Xent???????????????filter bank???????????????????????????????????????????????????????????????filter bank?????????$h$???????????????????????????????????????

      $\ell_{i}=-\log \frac{\exp \left(\operatorname{sim}\left(h_{i}, h_{j}\right) / \tau\right)}{\sum_{k=1}^{2 N} 1_{[k \neq i]} \exp \left(\operatorname{sim}\left(h_{i}, h_{k}\right) / \tau\right)}$
      $L_{\text {filter }}=\sum_{i=1}^{N} \ell_{i}$

3. Conditioning the generator to generate hybrid images.

   ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????? $I=G\left(x_{1}, y_{1}, z_{1}, b_{1}\right)$  $I_{\text {pos }_{2}}=G\left(x_{2}, y_{1}, z_{1}, b_{1}\right)$   ??????????????????NT-Xent?????????

   $L_{\text {hybrid }}=\sum_{i=1}^{N} \ell_{i}$

   

   $\mathcal{L}=L_{\text {base }}+L_{\text {hybrid }}+L_{\text {filter }}$

**experiments**

<img src="./image/image-20210603204635818.png" alt="image-20210603204635818" style="zoom:50%;" />

![image-20210603204650798](./image/image-20210603204650798.png)

![image-20210603204711347](https://data-1306015739.cos.ap-nanjing.myqcloud.com/image/image-20210603204711347.png)

