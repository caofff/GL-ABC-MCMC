# Summary

ABC-GL-MCMC is a python project of a recently proposed adaptive ABC-MCMC with Global-Local proposals (GL-ABC-MCMC) that targets for complex posterior distributions with intractable likelihood functions. 

--------
# Main File Descriptions
- GlobalMCMC.py: Implements ABC-MCMC with Global-Local proposals.
- GLMCMC.py: Implements the ABC-GL-MCMC with i-SIR as the global proposal and ordinary ABC-MCMC as local proposal.
- GLMALA.py: Implements the ABC-GL-MCMC with i-SIR as the global proposal and combining ABC and Metropolis-Adjusted Langevin Algorithm (MALA) as local proposal.
- GLMCMC_NFs.py: Implements the GL-MCMC algorithm that utilizes Normalizing Flows to improve importance proposals of i-SIR.
- Mixabs.py: Defines a mixture distribution model Mixture_set and provides example usage.
- Mixabs_GLMALA_bingxing.py: Runs the GL-MALA algorithm in parallel using multiprocessing.
- Mixabs_bingxing.py: Runs the GL-MCMC algorithm in parallel using multiprocessing.
--------
# Installation Dependencies
The project depends on the following Python packages:
- torch
- tqdm
- numpy
- matplotlib
- normflows
--------
# Parameter Descriptions
- num_ite: Number of iterations.
- Initial_theta: Initial parameters.
- Initial_y: Initial samples.
- Local_Proposal: Local proposal distribution.
- Global_Proposal: Global proposal distribution.
- filelocation: Path to save posterior samples.
- global_frequency: Frequency of global sampling.
- batch_size: Batch size.
- tau: Temperature parameter (for GL-MALA).
- num_grad: Number of samples for gradient estimation (for GL-MALA).
--------  
# Notes

Ensure all file paths are correct. Adjust parameter settings according to your needs. When running in parallel, ensure sufficient system resources.

--------
# Output
Posterior samples are saved in filelocation.

--------
# Demo

We refer users to run 'code/Mixabs.py ', 'code/Mixabs_GLMALA_bingxing.py' or 'code/Mixabs_bingxing.py' for an example.
