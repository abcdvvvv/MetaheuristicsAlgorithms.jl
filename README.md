# MetaheuristicsAlgorithms

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abdelazimhussien.github.io/MetaheuristicsAlgorithms.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abdelazimhussien.github.io/MetaheuristicsAlgorithms.jl/dev/)
[![Build Status](https://github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/abdelazimhussien/MetaheuristicsAlgorithms.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/abdelazimhussien/MetaheuristicsAlgorithms.jl.svg?branch=main)](https://app.travis-ci.com/abdelazimhussien/MetaheuristicsAlgorithms.jl)
<!-- [![Coverage](https://codecov.io/gh/abdelazimhussien/MetaheuristicsAlgorithms.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/abdelazimhussien/MetaheuristicsAlgorithms.jl) -->
[![Coverage](https://coveralls.io/repos/github/abdelazimhussien/MetaheuristicsAlgorithms.jl/badge.svg?branch=main)](https://coveralls.io/github/abdelazimhussien/MetaheuristicsAlgorithms.jl?branch=main)

## Introduction
MetaheuristicsAlgorithms.jl is a versatile Julia package that brings together a large collection of metaheuristic optimization algorithms in one place. It is designed to help users quickly apply and compare nature-inspired optimization methods on a variety of challenging problems.

From academic research to real-world engineering design, this package offers:

- Easy-to-use implementations of over 100 metaheuristic algorithms.

- Benchmark functions including classical engineering problems and standard test suites.

- Tools for analyzing algorithm performance and convergence.

- A clean and extensible code structure for customization.

Whether you’re exploring new optimization strategies or solving complex design tasks, MetaheuristicsAlgorithms.jl makes it simple and efficient.
<!-- MetaheuristicsAlgorithms.jl is one of the world’s most comprehensive Julia packages, offering an extensive collection of advanced metaheuristic algorithms. This package includes a diverse array of algorithms, meticulously organized in alphabetical order for ease of navigation: -->
<!-- ### A
- **<u>Artificial Electric Field Algorithm</u>**: Yadav, Anupam. "AEFA: Artificial electric field algorithm for global optimization." Swarm and Evolutionary Computation 48 (2019): 93-108.
- **<u>Artificial Ecosystem-based Optimization</u>**: Zhao, Weiguo, Liying Wang, and Zhenxing Zhang.  "Artificial ecosystem-based optimization: a novel nature-inspired meta-heuristic algorithm." Neural Computing and Applications 32, no. 13 (2020): 9383-9425.
- **<u>Ali Baba and the Forty Thieves</u>**: Braik, Malik, Mohammad Hashem Ryalat, and Hussein Al-Zoubi. "A novel meta-heuristic algorithm for solving numerical optimization problems: Ali Baba and the forty thieves." Neural Computing and Applications 34, no. 1 (2022): 409-455.
- **<u>Artificial Hummingbird Algorithm</u>**: Zhao, Weiguo, Liying Wang, and Seyedali Mirjalili. "Artificial hummingbird algorithm: A new bio-inspired optimizer with its engineering applications." Computer Methods in Applied Mechanics and Engineering 388 (2022): 114194.
- **<u>Artificial lemming algorithm </u>**: Xiao, Y., Cui, H., Khurma, R. A., & Castillo, P. A. (2025). 
Artificial lemming algorithm: a novel bionic meta-heuristic technique for solving real-world engineering optimization problems. 
Artificial Intelligence Review, 58(3), 84.
- **<u>ant lion optimizer</u>**: Mirjalili, Seyedali. "The ant lion optimizer." Advances in engineering software 83 (2015): 80-98.
- **<u>Arithmetic Optimization Algorithm</u>**: Abualigah, Laith, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, and Amir H. Gandomi. "The arithmetic optimization algorithm." Computer methods in applied mechanics and engineering 376 (2021): 113609.
- **<u>Artificial Protozoa Optimizer</u>**: Wang, Xiaopeng, Václav Snášel, Seyedali Mirjalili, Jeng-Shyang Pan, Lingping Kong, and Hisham A. Shehadeh. "Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization." Knowledge-Based Systems 295 (2024): 111737.
- **<u>Artificial Rabbits Optimization</u>**: Wang, Liying, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili, and Weiguo Zhao. "Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems." Engineering Applications of Artificial Intelligence 114 (2022): 105082.
- Yuan, Chong, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, Zongda Wu, and Huiling Chen. 
"Artemisinin optimization based on malaria therapy: Algorithm and applications to medical image segmentation." 
Displays 84 (2024): 102740.
- **<u>African Vultures Optimization Algorithm</u>**: Abdollahzadeh, B., Gharehchopogh, F. S., & Mirjalili, S. (2021). African vultures optimization algorithm: A new nature-inspired metaheuristic algorithm for global optimization problems. Computers & Industrial Engineering, 158, 107408.
### B
- **<u>Bald Eagle Search</u>**: Alsattar, Hassan A., A. A. Zaidan, and B. B. Zaidan. "Novel meta-heuristic bald eagle search optimisation algorithm." Artificial Intelligence Review 53 (2020): 2237-2264.
- **<u>Black-winged kite algorithm</u>**: Wang, Jun, Wen-chuan Wang, Xiao-xue Hu, Lin Qiu, and Hong-fei Zang. 
"Black-winged kite algorithm: a nature-inspired meta-heuristic for solving benchmark functions and engineering problems." 
Artificial Intelligence Review 57, no. 4 (2024): 98.
- **<u>Bonobo optimizer</u>**: Das, Amit Kumar, and Dilip Kumar Pratihar. 
"Bonobo optimizer (BO): an intelligent heuristic with self-adjusting parameters over continuous spaces and its applications to engineering problems." 
Applied Intelligence 52, no. 3 (2022): 2942-2974.
- **<u>Butterfly optimization algorithm</u>**: Arora, Sankalap, and Satvir Singh. 
"Butterfly optimization algorithm: a novel approach for global optimization." 
Soft computing 23 (2019): 715-734.
### C
- Braik, Malik, Alaa Sheta, and Heba Al-Hiary. "A novel meta-heuristic search algorithm for solving optimization problems: capuchin search algorithm." Neural computing and applications 33, no. 7 (2021): 2515-2547.
- **<u>Chernobyl disaster optimizer</u>**: Shehadeh, Hisham A. "Chernobyl disaster optimizer (CDO): A novel meta-heuristic method for global optimization." Neural Computing and Applications 35, no. 15 (2023): 10733-10749.
- **<u>Chameleon Swarm Algorithm</u>**: Braik, Malik Shehadeh. "Chameleon Swarm Algorithm: A bio-inspired optimizer for solving engineering design problems." Expert Systems with Applications 174 (2021): 114685.
- **<u>Chimp optimization algorithm</u>**: Khishe, Mohammad, and Mohammad Reza Mosavi. 
"Chimp optimization algorithm." 
Expert systems with applications 149 (2020): 113338.
- **<u>Cheetah Optimizer</u>**: Akbari, Mohammad Amin, Mohsen Zare, Rasoul Azizipanah-Abarghooee, Seyedali Mirjalili, and Mohamed Deriche. 
"The cheetah optimizer: A nature-inspired metaheuristic algorithm for large-scale optimization problems." 
Scientific reports 12, no. 1 (2022): 10953.
- **<u>Capuchin Search Algorithm</u>**: Braik, Malik, Alaa Sheta, and Heba Al-Hiary. 
"A novel meta-heuristic search algorithm for solving optimization problems: capuchin search algorithm." 
Neural computing and applications 33, no. 7 (2021): 2515-2547.
- **<u>COOT bird</u>**: Naruei, Iraj, and Farshid Keynia. 
"A new optimization method based on COOT bird natural life model." 
Expert Systems with Applications 183 (2021): 115352.
- **<u>Circulatory System Based Optimization</u>**: Ghasemi, Mojtaba, Mohammad-Amin Akbari, Changhyun Jun, Sayed M. Bateni, Mohsen Zare, Amir Zahedi, Hao-Ting Pai, 
Shahab S. Band, Massoud Moslehpour, and Kwok-Wing Chau. 
"Circulatory System Based Optimization (CSBO): an expert multilevel biologically inspired meta-heuristic algorithm." 
Engineering Applications of Computational Fluid Mechanics 16, no. 1 (2022): 1483-1525.
### D
- **<u>Dung Beetle Optimizer</u>**: Xue J, Shen B. 
Dung beetle optimizer: A new meta-heuristic algorithm for global optimization. 
The Journal of Supercomputing. 
2023 May;79(7):7305-36.
- **<u>Dynamic Differential Annealed Optimization</u>**: Ghafil, H. N., & Jármai, K. (2020). 
Dynamic differential annealed optimization: New metaheuristic optimization algorithm for engineering applications. 
Applied Soft Computing, 93, 106392.
- **<u>Dwarf Mongoose Optimization Algorithm</u>**: Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah. 
"Dwarf mongoose optimization algorithm." 
Computer methods in applied mechanics and engineering 391 (2022): 114570.
- **<u>Dandelion Optimizer</u>**: Zhao, Shijie, Tianran Zhang, Shilin Ma, and Miao Chen. 
"Dandelion Optimizer: A nature-inspired metaheuristic algorithm for engineering applications." 
Engineering Applications of Artificial Intelligence 114 (2022): 105075.
-  **<u>Deep Sleep Optimiser</u>**: Oladejo, Sunday O., Stephen O. Ekwe, Lateef A. Akinyemi, and Seyedali A. Mirjalili. 
"The deep sleep optimiser: A human-based metaheuristic approach." 
IEEE Access (2023).
### E
- Lian, Junbo, Ting Zhu, Ling Ma, Xincan Wu, Ali Asghar Heidari, Yi Chen, Huiling Chen, and Guohua Hui. 
"The educational competition optimizer." 
International Journal of Systems Science 55, no. 15 (2024): 3185-3222.
- Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash. 
"Exponential distribution optimizer (EDO): a novel math-inspired algorithm for global optimization and engineering problems." 
Artificial Intelligence Review 56, no. 9 (2023): 9329-9400.
- Al-Betar, M.A., Awadallah, M.A., Braik, M.S. et al. 
Elk herd optimizer: a novel nature-inspired metaheuristic algorithm. 
Artif Intell Rev 57, 48 (2024). 
https://doi.org/10.1007/s10462-023-10680-4
- **<u>Escape Algorithm (ESC) </u>**: "Ouyang, K., Fu, S., Chen, Y., Cai, Q., Heidari, A. A., & Chen, H. (2024). 
Escape: an optimization method based on crowd evacuation behaviors. 
Artificial Intelligence Review, 58(1), 19."
- Faramarzi, Afshin, Mohammad Heidarinejad, Brent Stephens, and Seyedali Mirjalili. 
"Equilibrium optimizer: A novel optimization algorithm." 
Knowledge-based systems 191 (2020): 105190.
- Luan, Tran Minh, Samir Khatir, Minh Thi Tran, Bernard De Baets, and Thanh Cuong-Le. 
"Exponential-trigonometric optimization algorithm for solving complicated engineering problems." 
Computer Methods in Applied Mechanics and Engineering 432 (2024): 117411.
### F
- Qi, Ailiang, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, and Huiling Chen. 
"FATA: an efficient optimization method based on geophysics." 
Neurocomputing 607 (2024): 128289. https://doi.org/10.1016/j.neucom.2024.128289 
- Hashim, Fatma A., Reham R. Mostafa, Abdelazim G. Hussien, Seyedali Mirjalili, and Karam M. Sallam. 
"Fick’s Law Algorithm: A physical law-based algorithm for numerical optimization." 
Knowledge-Based Systems 260 (2023): 110146.
- Ghasemi, M., Golalipour, K., Zare, M., Mirjalili, S., Trojovský, P., Abualigah, L. and Hemmati, R., 2024. 
Flood algorithm (FLA): an efficient inspired meta-heuristic for engineering optimization. 
The Journal of Supercomputing, 80(15), pp.22913-23017.
- Mohammed, Hardi, and Tarik Rashid. 
"FOX: a FOX-inspired optimization algorithm." 
Applied Intelligence 53, no. 1 (2023): 1030-1050.
### G
- Agushaka, Jeffrey O., Absalom E. Ezugwu, and Laith Abualigah. 
"Gazelle optimization algorithm: a novel nature-inspired metaheuristic optimizer." 
Neural Computing and Applications 35, no. 5 (2023): 4099-4131.
- Ahmadianfar, Iman, Omid Bozorg-Haddad, and Xuefeng Chu. 
"Gradient-based optimizer: A new metaheuristic optimization algorithm." 
Information Sciences 540 (2020): 131-159.
- Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Mohammad-Amin Akbari, Seyedali Mirjalili, and Laith Abualigah. 
"Geyser inspired algorithm: a new geological-inspired meta-heuristic for real-parameter and constrained engineering optimization." 
Journal of Bionic Engineering 21, no. 1 (2024): 374-408.
- El-Kenawy, El-Sayed M., Nima Khodadadi, Seyedali Mirjalili, Abdelaziz A. Abdelhamid, Marwa M. Eid, and Abdelhameed Ibrahim. 
"Greylag goose optimization: nature-inspired optimization algorithm." 
Expert Systems with Applications 238 (2024): 122147.
- Chopra, Nitish, and Muhammad Mohsin Ansari. 
"Golden jackal optimization: A novel nature-inspired optimizer for engineering applications." 
Expert Systems with Applications 198 (2022): 116924.
- Hu, Gang, Yuxuan Guo, Guo Wei, and Laith Abualigah. 
"Genghis Khan shark optimizer: a novel nature-inspired algorithm for engineering optimization." 
Advanced Engineering Informatics 58 (2023): 102210.
- Zhang, Yiying, Zhigang Jin, and Seyedali Mirjalili. 
"Generalized normal distribution optimization and its applications in parameter extraction of photovoltaic models." 
Energy Conversion and Management 224 (2020): 113301.
- Zhang, Qingke, Hao Gao, Zhi-Hui Zhan, Junqing Li, and Huaxiang Zhang. 
"Growth Optimizer: A powerful metaheuristic algorithm for solving continuous and discrete global optimization problems." 
Knowledge-Based Systems 261 (2023): 110206.
- Saremi, Shahrzad, Seyedali Mirjalili, and Andrew Lewis. 
"Grasshopper optimisation algorithm: theory and application." 
Advances in engineering software 105 (2017): 30-47.
- Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, and Seyedali Mirjalili. 
"Artificial gorilla troops optimizer: a new nature‐inspired metaheuristic algorithm for global optimization problems." 
International Journal of Intelligent Systems 36, no. 10 (2021): 5887-5958.
- Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. 
"Grey wolf optimizer." 
Advances in engineering software 69 (2014): 46-61.
### H
- Hashim, Fatma A., Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, and Walid Al-Atabany. 
"Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems." 
Mathematics and Computers in Simulation 192 (2022): 84-110.
- Lian, Junbo, and Guohua Hui. "Human evolutionary optimization algorithm." Expert Systems with Applications 241 (2024): 122638.
- Askari, Qamar, Mehreen Saeed, and Irfan Younas. 
"Heap-based optimizer inspired by corporate rank hierarchy for global optimization." 
Expert Systems with Applications 161 (2020): 113702.
- Yang, Yutao, Huiling Chen, Ali Asghar Heidari, and Amir H. Gandomi. 
"Hunger games search: Visions, conception, implementation, deep analysis, perspectives, and towards performance shifts." 
Expert Systems with Applications 177 (2021): 114864.
- Hashim, F.A., Houssein, E.H., Mabrouk, M.S., Al-Atabany, W. and Mirjalili, S., 2019. 
Henry gas solubility optimization: A novel physics-based algorithm. 
Future Generation Computer Systems, 101, pp.646-667.
- Heidari, Ali Asghar, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, and Huiling Chen. 
"Harris hawks optimization: Algorithm and applications." 
Future generation computer systems 97 (2019): 849-872.
- Oladejo, Sunday O., Stephen O. Ekwe, and Seyedali Mirjalili. 
"The Hiking Optimization Algorithm: A novel human-based metaheuristic approach." 
Knowledge-Based Systems 296 (2024): 111880.
- Amiri, M.H., Mehrabi Hashjin, N., Montazeri, M., Mirjalili, S. and Khodadadi, N., 2024. 
Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm. 
Scientific Reports, 14(1), p.5032.
- MiarNaeimi, Farid, Gholamreza Azizyan, and Mohsen Rashki. 
"Horse herd optimization algorithm: A nature-inspired algorithm for high-dimensional optimization problems." 
Knowledge-Based Systems 213 (2021): 106711.
### I
- Ahmadianfar, Iman, Ali Asghar Heidari, Saeed Noshadian, Huiling Chen, and Amir H. Gandomi. 
"INFO: An efficient optimization algorithm based on weighted mean of vectors." 
Expert Systems with Applications 195 (2022): 116516.
- Ghasemi, Mojtaba, Mohsen Zare, Pavel Trojovský, Ravipudi Venkata Rao, Eva Trojovská, and Venkatachalam Kandasamy. 
"Optimization based on the smart behavior of plants with its engineering applications: Ivy algorithm." 
Knowledge-Based Systems 295 (2024): 111850.
### J
- Rao, R. 
"Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems." 
International Journal of Industrial Engineering Computations 7, no. 1 (2016): 19-34.
- Chou, Jui-Sheng, and Dinh-Nhat Truong. 
"A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean." 
Applied Mathematics and Computation 389 (2021): 125535.
### L
- Houssein, Essam H., Diego Oliva, Nagwan Abdel Samee, Noha F. Mahmoud, and Marwa M. Emam. 
"Liver Cancer Algorithm: A novel bio-inspired optimizer." 
Computers in Biology and Medicine 165 (2023): 107389.
- Houssein, Essam H., Mohammed R. Saad, Fatma A. Hashim, Hassan Shaban, and M. Hassaballah. 
"Lévy flight distribution: A new metaheuristic algorithm for solving engineering optimization problems." 
Engineering Applications of Artificial Intelligence 94 (2020): 103731.
- Ghasemi, Mojtaba, Mohsen Zare, Amir Zahedi, Pavel Trojovský, Laith Abualigah, and Eva Trojovská. 
"Optimization based on performance of lungs in body: Lungs performance-based optimization (LPO)." 
Computer Methods in Applied Mechanics and Engineering 419 (2024): 116582.
### M
- Zheng, Boli, Yi Chen, Chaofan Wang, Ali Asghar Heidari, Lei Liu, and Huiling Chen. 
"The moss growth optimization (MGO): concepts and performance." 
Journal of Computational Design and Engineering 11, no. 5 (2024): 184-221.
- Abdollahzadeh, Benyamin, Farhad Soleimanian Gharehchopogh, Nima Khodadadi, and Seyedali Mirjalili. 
"Mountain gazelle optimizer: a new nature-inspired metaheuristic algorithm for global optimization problems." 
Advances in Engineering Software 174 (2022): 103282.
- Faramarzi, Afshin, Mohammad Heidarinejad, Seyedali Mirjalili, and Amir H. Gandomi. 
"Marine Predators Algorithm: A nature-inspired metaheuristic." 
Expert systems with applications 152 (2020): 113377.
- Zhao, Weiguo, Zhenxing Zhang, and Liying Wang. 
"Manta ray foraging optimization: An effective bio-inspired optimizer for engineering applications." 
Engineering Applications of Artificial Intelligence 87 (2020): 103300.
- Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Abdolreza Hatamlou. 
"Multi-verse optimizer: a nature-inspired algorithm for global optimization." 
Neural Computing and Applications 27 (2016): 495-513.
### O
- Dehghani, Mohammad, and Pavel Trojovský. 
"Osprey optimization algorithm: A new bio-inspired metaheuristic algorithm for solving engineering optimization problems." 
Frontiers in Mechanical Engineering 8 (2023): 1126450.
### P
- Lian, Junbo, Guohua Hui, Ling Ma, Ting Zhu, Xincan Wu, Ali Asghar Heidari, Yi Chen, and Huiling Chen. 
"Parrot optimizer: Algorithm and applications to medical problems." 
Computers in Biology and Medicine 172 (2024): 108064.
- Ezugwu, Absalom E., Jeffrey O. Agushaka, Laith Abualigah, Seyedali Mirjalili, and Amir H. Gandomi. 
"Prairie dog optimization algorithm." 
Neural Computing and Applications 34, no. 22 (2022): 20017-20065.
- Bouaouda, Anas, Fatma A. Hashim, Yassine Sayouti, and Abdelazim G. Hussien. 
"Pied kingfisher optimizer: a new bio-inspired algorithm for solving numerical optimization and industrial engineering problems." 
Neural Computing and Applications (2024): 1-59.
- Yuan, Chong, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, and Huiling Chen. "Polar lights optimizer: Algorithm and applications in image segmentation and feature selection." Neurocomputing 607 (2024): 128427.
- Trojovský, Pavel, and Mohammad Dehghani. "Pelican optimization algorithm: A novel nature-inspired algorithm for engineering applications." Sensors 22, no. 3 (2022): 855.
- Askari, Qamar, Irfan Younas, and Mehreen Saeed. 
"Political Optimizer: A novel socio-inspired meta-heuristic for global optimization." 
Knowledge-based systems 195 (2020): 105709.
- Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri. 
"Poor and rich optimization algorithm: A new human-based and multi populations algorithm." 
Engineering applications of artificial intelligence 86 (2019): 165-181.
- Abdollahzadeh, Benyamin, Nima Khodadadi, Saeid Barshandeh, Pavel Trojovský, Farhad Soleimanian Gharehchopogh, El-Sayed M. El-kenawy, Laith Abualigah, and Seyedali Mirjalili. 
"Puma optimizer (PO): A novel metaheuristic optimization algorithm and its application in machine learning." 
Cluster Computing (2024): 1-49.
### Q
- Zhao, Weiguo, Liying Wang, Zhenxing Zhang, Seyedali Mirjalili, Nima Khodadadi, and Qiang Ge. 
"Quadratic Interpolation Optimization (QIO): A new optimization algorithm based on generalized quadratic interpolation and its applications to real-world engineering problems." 
Computer Methods in Applied Mechanics and Engineering 417 (2023): 116446.
### R
- **<u>Red-billed blue magpie optimizer (RBMO)</u>**: Fu, Shengwei, et al. 
"Red-billed blue magpie optimizer: a novel metaheuristic algorithm for 2D/3D UAV path planning and engineering design problems." 
Artificial Intelligence Review 57.6 (2024): 134.
- Su, Hang, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, and Huiling Chen. "RIME: A physics-based optimization." Neurocomputing 532 (2023): 183-214.
- Jia, Heming, Xiaoxu Peng, and Chunbo Lang. 
"Remora optimization algorithm." 
Expert Systems with Applications 185 (2021): 115665.
- Abualigah, Laith, Mohamed Abd Elaziz, Putra Sumari, Zong Woo Geem, and Amir H. Gandomi. 
"Reptile Search Algorithm (RSA): A nature-inspired meta-heuristic optimizer." 
Expert Systems with Applications 191 (2022): 116158.
- Dhiman, Gaurav, Meenakshi Garg, Atulya Nagar, Vijay Kumar, and Mohammad Dehghani. 
"A novel algorithm for global optimization: rat swarm optimizer." 
Journal of Ambient Intelligence and Humanized Computing 12 (2021): 8457-8482.
- Ahmadianfar, Iman, Ali Asghar Heidari, Amir H. Gandomi, Xuefeng Chu, and Huiling Chen. 
"RUN beyond the metaphor: An efficient optimization algorithm based on Runge Kutta method." 
Expert Systems with Applications 181 (2021): 115079.
### S
- Moosavi, Seyyed Hamid Samareh, and Vahid Khatibi Bardsiri. 
"Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation." 
Engineering Applications of Artificial Intelligence 60 (2017): 1-15.
- Fu, Youfa, Dan Liu, Jiadui Chen, and Ling He. 
"Secretary bird optimization algorithm: a new metaheuristic for solving global optimization problems." 
Artificial Intelligence Review 57, no. 5 (2024): 1-102.
- **<u>Sine Cosine Algorithm</u>**: Mirjalili, Seyedali. "SCA: a sine cosine algorithm for solving optimization problems."Knowledge-based systems 96 (2016): 120-133.
- Bai, Jianfu, Yifei Li, Mingpo Zheng, Samir Khatir, Brahim Benaissa, Laith Abualigah, and Magd Abdel Wahab. 
"A sinh cosh optimizer." 
Knowledge-Based Systems 282 (2023): 111081.
- **<u>Starfish Optimization Algorithm (SFOA)</u>**: Zhong, C., Li, G., Meng, Z., Li, H., Yildiz, A. R., & Mirjalili, S. (2025). 
"Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers. "
Neural Computing and Applications, 37(5), 3641-3683.
- Özbay, Feyza Altunbey. 
"A modified seahorse optimization algorithm based on chaotic maps for solving global optimization and engineering problems." 
Engineering Science and Technology, an International Journal 41 (2023): 101408.
- Dhiman, Gaurav, and Vijay Kumar. 
"Spotted hyena optimizer: a novel bio-inspired based metaheuristic technique for engineering applications." 
Advances in Engineering Software 114 (2017): 48-70.
- Li, Shimin, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, and Seyedali Mirjalili. 
"Slime mould algorithm: A new method for stochastic optimization." 
Future generation computer systems 111 (2020): 300-323.
- Deng, Lingyun, and Sanyang Liu. 
"Snow ablation optimizer: A novel metaheuristic technique for numerical optimization and engineering design." 
Expert Systems with Applications 225 (2023): 120069.
- Hashim, Fatma A., and Abdelazim G. Hussien. 
"Snake Optimizer: A novel meta-heuristic optimization algorithm." 
Knowledge-Based Systems 242 (2022): 108320.
- Dhiman, Gaurav, and Vijay Kumar. "Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems." Knowledge-based systems 165 (2019): 169-196.
- Jiankai Xue & Bo Shen 
A novel swarm intelligence optimization approach: sparrow search algorithm
Systems Science & Control Engineering, 8:1, 22-34, 
DOI: 10.1080/21642583.2019.1708830
(2020) 
- Mirjalili, Seyedali, Amir H. Gandomi, Seyedeh Zahra Mirjalili, Shahrzad Saremi, Hossam Faris, and Seyed Mohammad Mirjalili. 
"Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems." 
Advances in engineering software 114 (2017): 163-191.
- Dhiman, Gaurav, and Amandeep Kaur. 
"STOA: a bio-inspired based optimization algorithm for industrial engineering problems." 
Engineering Applications of Artificial Intelligence 82 (2019): 148-174.
- **<u>Superb Fairy-wren Optimization Algorithm (SuperbFOA)</u>** Jia, Heming, et al. 
"Superb Fairy-wren Optimization Algorithm: a novel metaheuristic algorithm for solving feature selection problems." 
Cluster Computing 28.4 (2025): 246.
- **<u>Supply-demand-based optimization (SDO)</u>**  Zhao, Weiguo, Liying Wang, and Zhenxing Zhang. 
"Supply-demand-based optimization: A novel economics-inspired algorithm for global optimization." 
Ieee Access 7 (2019): 73182-73206.
### T
- Rao, R. Venkata, Vimal J. Savsani, and Dipakkumar P. Vakharia. 
"Teaching–learning-based optimization: a novel method for constrained mechanical design optimization problems." 
Computer-aided design 43, no. 3 (2011): 303-315.
- Minh, Hoang-Le, Thanh Sang-To, Guy Theraulaz, Magd Abdel Wahab, and Thanh Cuong-Le. 
"Termite life cycle optimizer." 
Expert Systems with Applications 213 (2023): 119211.
- Kaur, Satnam, Lalit K. Awasthi, Amrit Lal Sangal, and Gaurav Dhiman. 
"Tunicate Swarm Algorithm: A new bio-inspired based metaheuristic paradigm for global optimization." 
Engineering Applications of Artificial Intelligence 90 (2020): 103541.
- Zhao, Shijie, Tianran Zhang, Liang Cai, and Ronghua Yang. 
"Triangulation topology aggregation optimizer: A novel mathematics-based meta-heuristic algorithm for 
continuous optimization and engineering applications." 
Expert Systems with Applications 238 (2024): 121744.
### W
- Naruei, Iraj, and Farshid Keynia. 
"Wild horse optimizer: A new meta-heuristic algorithm for solving engineering optimization problems." 
Engineering with computers 38, no. Suppl 4 (2022): 3025-3056.
- Han, Muxuan, Zunfeng Du, Kum Fai Yuen, Haitao Zhu, Yancang Li, and Qiuyu Yuan. 
"Walrus optimizer: A novel nature-inspired metaheuristic algorithm." 
Expert Systems with Applications 239 (2024): 122413.
- Mirjalili, Seyedali, and Andrew Lewis. 
"The whale optimization algorithm." 
Advances in engineering software 95 (2016): 51-67.
- Braik, Malik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar, and Mohammed A. Awadallah. 
"White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm for global optimization problems." 
Knowledge-Based Systems 243 (2022): 108457.
- Braik, M., & Al-Hiary, H. (2025). 
A novel meta-heuristic optimization algorithm inspired by water uptake and transport in plants. 
Neural Computing and Applications, 1-82.
### Y
- Abdel-Basset, Mohamed, Doaa El-Shahat, Mohammed Jameel, and Mohamed Abouhawwash.
"Young’s double-slit experiment optimizer: A novel metaheuristic optimization algorithm for global and constraint optimization problems." 
Computer Methods in Applied Mechanics and Engineering 403 (2023): 115652.
### Z
- Trojovská, Eva, Mohammad Dehghani, and Pavel Trojovský. 
"Zebra optimization algorithm: A new bio-inspired optimization algorithm for solving optimization algorithm." 
Ieee Access 10 (2022): 49445-49473.-->