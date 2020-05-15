%Author: Deng-Ping Fan
%Date: 2018-12-10
%Email:dengpingfan@mail.nankai.edu.cn
%Homepage: http://dpfan.net/
%This evaluation code is only a demo. You may get overall evaluation results from my homepage.
%If you have any questions, please don't hesitate to contact me.

%%%%%%%How to do?
1. go to the folder of 'Evaluation Code' and run the main.m. 
2. The results are stored in 'Result' folder.
3. If you want to test you own model. Please put your saliency map in the 'SalMap' folder similar like the LHM model structure.
    SalMap->LHM->NLPR->xxx.png (the same with ground truth's filename).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Note that if you use our code, please cite the following papers:
@inproceedings{fan2017structure,
	title={{Structure-measure: A New Way to Evaluate Foreground Maps}},
	author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
	booktitle={IEEE International Conference on Computer Vision (ICCV)},
	pages = {4548-4557},
	year={2017},
	note={\url{http://dpfan.net/smeasure/}},
	organization={IEEE}
}

@inproceedings{Fan2018Enhanced,
	author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
	title={{Enhanced-alignment Measure for Binary Foreground Map Evaluation}},
	booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
	pages={698--704},
	note={\url{http://dpfan.net/e-measure/}},
	year={2018}
}

@inproceedings{fan2018SOC,
	title={Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground},
	author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Jiang-Jiang and Gao, Shang-Hua and Hou, Qibin and Borji, Ali},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year={2018},
	organization={Springer}
}