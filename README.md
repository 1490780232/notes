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
**Problem:** 分层次生成细粒度的真实图片 Disentangle factors without supervision to generate realistic and diverse images for fine-grained classes.

**motivation:** 作者认为无监督的方法可以发现数据中隐藏的结构。细粒度的图片数据中包含内在的层次组织结构，可以先用一种粗粒度的类别生成大致的图片，然后用更细粒度的图片去做微调。再文章中，作者用父编码和子编码分别代表物体的形状和纹理特征。因为没有任何形状与纹理特征作为监督，作者采用父编码与物体形状之间的互信息来约束他们，使得他们能够一一对应，具体表现在一个编码对应一中形状或者纹理特征。

**Solution:**

FineGAN分不同阶段来生成细粒度的图片，如：第一个阶段生成背景图片，第二个阶段在背景上嵌入物体的形状，第三个阶段则继续在物体形状上添加纹理信息。为了将物体的形状和背景在无监督的情况下解耦开来，作者增强每一个编码和颜色或纹理之间的互信息（直接用分类损失去计算）让每个父编码和自编码代表一类颜色和纹理信息。图片生成的整理流程是先使用背景编码生成背景，接着使用父编码生成物体形状将物体嵌入到背景中，最后使用自编码给物体加上纹理特征。

**Hierarchical finegrained disentanglement**

![image-20210531094305068](./image/image-20210531094305068.png)

1. 输入: $z \sim \mathcal{N}(0,1)$ $b \sim \operatorname{Cat}\left(K=N_{b}, p=1 / N_{b}\right)$ $p \sim \operatorname{Cat}\left(K=N_{p}, p=1 / N_{p}\right)$ $c \sim \operatorname{Cat}\left(K=N_{c}, p=1 / N_{c}\right)$，这里b,p,c为onehot编码，$N_{b},N_{p},N_{c}$分别代表背景，形状，纹理的种类数量

2. 不同编码之间的关系: 对于每一个父编码，作者会设定固定数量的子编码与之对应，也就是说，会有几个子编码共享一个父编码，例如：如果一个父编码表示鸭子的形状，与之对应的子编码表示鸭子的不同的颜色和纹理

3. 背景生成阶段：

   1. 在这一阶段中，作者首先用一个物体检测模型去检测物体所在的位置，接着，将图片分成$N \times N$个patch，每个patch对应物体区域或者背景区域，然后，判别器$D_{b}$对每个patch进行判别，计算loss，生成器则生成背景图片，对抗损失为：

      $\mathcal{L}_{b g_{-} a d v}=\min _{G_{b}} \max _{D_{b}} \mathbb{E}_{x}\left[\log \left(D_{b}(x)\right)\right]+\mathbb{E}_{z, b} \mid\left[\log \left(1-D_{b}\left(G_{b}(z, b)\right)\right)\right]$

      除了对抗训练外，作者还先根据划分的背景和物体区域单独预训练了一个辅助的判别器$D_{aux}$，并使用这个判别器去训练$G_b$ 损失为：$\mathcal{L}_{b g_{-} a u x}=\min _{G_{b}} \mathbb{E}_{z, b}\left[\log \left(1-D_{a u x}\left(G_{b}(z, b)\right)\right)\right]$

4. 父阶段(形状生成阶段)

   1. 父阶段可以被认为给建模物体的高层特征，如形状，而子阶段则被认为建模物体更底层的特征比如颜色和纹理等

   2. 父阶段主要产生一个前景实体，并把它粘到背景图片$\mathcal{B}$上去，因此，在这阶段有两个生成器$G_{p, f}$和$G_{p, m}$去生成前景图$\mathcal{P}_{f}$和形状掩码$\mathcal{P}_{m}$，得到的形状图片为$\mathcal{P}=\mathcal{P}_{f, m}+\mathcal{B}_{m}$，其中 $\mathcal{P}_{f, m}=\mathcal{P}_{m} \odot \mathcal{P}_{f}$ ，$\mathcal{B}_{m}=\left(1-\mathcal{P}_{m}\right) \odot \mathcal{B}$

   3. 在这阶段，作者仅使用$D_{p}$ 去引导父编码表示物体的形状信息。因为没有标签，作者就像InfoGAN一样，去最大化父编码$p$和前景图片${P}_{f, m}$的互信息，然后父编码和前景图片联系起来。用作者的话来讲，就是$D_p$需要从生成器$G_{p, f}, G_{p, m}$生成的前景图中重构出父编码（形状信息）来。**具体的做法就是将generator生成的前景图， 经过一个分类器，看是否能够得到p,(父编码是onehot编码，每个编码代表一类)**

      因此，父阶段的损失函数为：$\mathcal{L}_{p}=\mathcal{L}_{p_{-} i n f o}=\max _{D_{p}, G_{p, f}, G_{p, m}} \mathbb{E}_{z, p}\left[\log D_{p}\left(p \mid \mathcal{P}_{f, m}\right)\right]$
      
      info的损失代码如下， criterion_class就是一个交叉熵损失，p_code表示父编码，pred_p则是discriminator预测的概率，在这一阶段，只需要优化这一个损失即可
      
      ```python
      if i == 1: # Mutual information loss for the parent stage (1)
              pred_p = self.netsD[i](self.fg_mk[i-1])
              errG_info = criterion_class(pred_p[0], torch.nonzero(p_code.long())[:,1])
      elif i == 2: # Mutual information loss for the child stage (2)
              pred_c = self.netsD[i](self.fg_mk[i-1])
              errG_info = criterion_class(pred_c[0], torch.nonzero(c_code.long())[:,1]
      ```
      
      

5. 子阶段

   1. 首先，子编码先和上一阶段产生的特征连接在一起输入到生成器$G_c$中，与父阶段一样，两个生成器$G_{c, f}$  $G_{c, m}$去生成物体的前景图和掩码图，将他们相乘，生成物体的图片，然后将该物体的图片贴到上一阶段的图片中，具体可表示为$\mathcal{C}=\mathcal{C}_{f, m}+\mathcal{P}_{c, m}$ $\mathcal{C}_{f, m}=\mathcal{C}_{m} \odot \mathcal{C}_{f}$, $\mathcal{P}_{c, m}=\left(1-\mathcal{C}_{m}\right) \odot \mathcal{P}$
   2. 这一阶段的loss可以分成两个，一个是与父阶段一致:用一个互信息的损失将子编码和纹理颜色特征相联系：$\mathcal{L}_{c_{-} i n f o}=\max _{D_{c}, G_{c, f}, G_{c, m}} \mathbb{E}_{z, p, c}\left[\log D_{c}\left(c \mid \mathcal{C}_{f, m}\right)\right]$，然后一个整体的对抗损失加在整张图片上，然后整个流程生成的图片更加真实$\mathcal{L}_{c_{-} a d v}=\min _{G_{c}} \max _{D_{a d v}} \mathbb{E}_{x}\left[\log \left(D_{a d v}(x)\right)\right]+\mathbb{E}_{z, b, p, c}\left[\log \left(1-D_{a d v}(\mathcal{C})\right)\right]$

   此外，作者根据生成的图片，训练了两个分类器去分类图片的父编码和子编码，并使用倒数第二层的特征去做聚类，取得了较号的效果

   **experiments**

   <img src="./image/image-20210603194536162.png" alt="image-20210603194536162" style="zoom:50%;" />

   <img src="./image/image-20210603194556218.png" alt="image-20210603194556218" style="zoom:50%;" />

   

   <img src="./image/image-20210603194610650.png" alt="image-20210603194610650" style="zoom:50%;" />



### GENERATING FURRY CARS: DISENTANGLING OBJECT SHAPE & APPEARANCE ACROSS MULTIPLE DOMAINS

**Paper name:** ENERATING FURRY CARS: DISENTANGLING OBJECT SHAPE & APPEARANCE ACROSS MULTIPLE DOMAINS
**Publication:** iclr2021
**Problem:** gan
**motivation:** 这篇论文考虑了将不同域之间的物体形状和外观特征给解耦出来，比如（狗和车两个领域。之前的方法可以解耦出一个域的物体但是跨度太大的领域效果可能不太行。改方法最主要的难点在于如何精确地将一个物体的形状，外观和背景解耦出来，以方便不同领域之间相互利用。本文的主要贡献在于将物体的外表用一个可导的视觉特征来表示，并优化生成器，使得具有相同内容外观但是不同形状的特征可以产生相同的视觉直方图。

**method:**

<img src="./image/image-20210603200106268.png" alt="image-20210603200106268" style="zoom: 50%;" />

<img src="./image/image-20210603195033365.png" alt="image-20210603195033365" style="zoom:50%;" />

1. FineGAN:
   
   1. 与之前上一篇的结构一样，这里不过多阐述，但之前的每一个父编码e都会对应几个相应的子编码，如果使用别的父编码对应的子编码生成的图片可能不太好，因此，本文则为了让域之间的颜色纹理特能共享，提取了一个底层特征并结合对比学习去优化网络
   
2. Learning a differentiable histogram of low-level features.

   1. 作者主要想让部拥有不同形状的物体有着相同的外观特征，为了把这个想法加入到优化过程中去，首先需要设计如何提取底层特征变更让他可到，其次需要引导生成器在生成混合图片时保存底层特征

   2. 为了得到物体的颜色纹理特征，作者首先用一组卷积（filter bank）去提取图片特征，然后与对应的mask相乘，得到物体区域的特征，在通道维度上进行求和，得到$k$维的直方图$h$，作者使用这$k$维的直方图去表示底层特征

   3. 为了保证直方图$h$得到的确实是某些物体的特征，作者采用了对比学习的方式，首先构建一批图片，对于其中的一张图片$I=G\left(x_{1}, y_{1}, z_{1}, b_{1}\right)$，变化$z$得到不同pose的正样本$I_{\text {pos }_{1}}=G\left(x_{1}, y_{1}, z_{2}, b_{1}\right)$，这两张图片的纹理和形状和颜色相同,接着，使用NT-Xent损失去优化filter bank，保持保持网络中其他的权重不变，通过这样让filter bank得到的$h$能代表物体的颜色和纹理特征

      $\ell_{i}=-\log \frac{\exp \left(\operatorname{sim}\left(h_{i}, h_{j}\right) / \tau\right)}{\sum_{k=1}^{2 N} 1_{[k \neq i]} \exp \left(\operatorname{sim}\left(h_{i}, h_{k}\right) / \tau\right)}$
      $L_{\text {filter }}=\sum_{i=1}^{N} \ell_{i}$

3. Conditioning the generator to generate hybrid images.

   对于不同域之间的图片，作者为了让不同形状的物体能够共享同一个纹理特征，在不同形状物体之间构建了正负样本 $I=G\left(x_{1}, y_{1}, z_{1}, b_{1}\right)$  $I_{\text {pos }_{2}}=G\left(x_{2}, y_{1}, z_{1}, b_{1}\right)$   接着同样使用NT-Xent去优化

   $L_{\text {hybrid }}=\sum_{i=1}^{N} \ell_{i}$

   

   $\mathcal{L}=L_{\text {base }}+L_{\text {hybrid }}+L_{\text {filter }}$

**experiments**

<img src="./image/image-20210603204635818.png" alt="image-20210603204635818" style="zoom:50%;" />

![image-20210603204650798](./image/image-20210603204650798.png)

![image-20210603204711347](https://data-1306015739.cos.ap-nanjing.myqcloud.com/image/image-20210603204711347.png)

