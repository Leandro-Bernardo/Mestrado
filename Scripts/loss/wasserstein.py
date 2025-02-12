
import torch
import pytorch_lightning as pl
from typing import Optional, List

class Wasserstein(pl.LightningModule):
    """
    Compute the  Wasserstein distance between two 1D discrete distributions with momentum p.

    The Wasserstein distance is a similarity metric between two probability
    distributions . In the discrete case, the Wasserstein distance can be
    understood as the cost of an optimal transport plan to convert one
    distribution into the other. The cost is calculated as the product of the
    amount of probability mass being moved and the distance it is being moved.

    When p=1, the Wasserstein distance is equivalent to the Earth Mover's Distance


    Args:
        u_values : 1d torch.Tensor
            A sample from a probability distribution or the support (set of all
            possible values) of a probability distribution. Each element is an
            observation or possible value.

        v_values : 1d torch.Tensor
            A sample from or the support of a second distribution.

    Returns:
        distance : torch.tensor
            The computed distance between the distributions.
    """
    def __init__(self, u_values: torch.Tensor = None, v_values: torch.Tensor = None, p: int = 1):
        super().__init__()


    def forward(self, u_values: torch.Tensor, v_values: torch.Tensor, u_weights: Optional[torch.Tensor] = None, v_weights: Optional[torch.Tensor] = None, p: int =1):

        # X-Y
        Z = u_values - v_values

        # CDF(X-Y) (equivalent to CDF(X) - CDF(Y))
        cumsum = torch.cumsum(Z,dim=0)

        # |(CDF(X-Y))|
        abs_cumsum = torch.abs(cumsum)

        # W(X,Y) = Σ|(CDF(X-Y))|
        wasserstein_distance = torch.sum(abs_cumsum)

        return wasserstein_distance



# class Wasserstein(pl.LightningModule):
#     """
#     Compute the  Wasserstein distance between two 1D discrete distributions with momentum p.

#     The Wasserstein distance is a similarity metric between two probability
#     distributions . In the discrete case, the Wasserstein distance can be
#     understood as the cost of an optimal transport plan to convert one
#     distribution into the other. The cost is calculated as the product of the
#     amount of probability mass being moved and the distance it is being moved.

#     When p=1, the Wasserstein distance is equivalent to the Earth Mover's Distance


#     Args:
#         u_values : 1d torch.Tensor
#             A sample from a probability distribution or the support (set of all
#             possible values) of a probability distribution. Each element is an
#             observation or possible value.

#         v_values : 1d torch.Tensor
#             A sample from or the support of a second distribution.

#         u_weights, v_weights : 1d array_like, optional
#             Weights or counts corresponding with the sample or probability masses
#             corresponding with the support values. Sum of elements must be positive
#             and finite. If unspecified, each value is assigned the same weight.

#     Returns:
#         distance : torch.tensor
#             The computed distance between the distributions.
#     """
#     def __init__(self, u_values: torch.Tensor = None, v_values: torch.Tensor = None, u_weights: Optional[torch.Tensor] = None, v_weights: Optional[torch.Tensor] = None, p: int = 1):
#         super().__init__()
#         # self.u_values = u_values
#         # self.v_values = v_values
#         # self.u_weights = u_weights
#         # self.v_weights = v_weights
#         # self.p = p

#         #assert self.u_values != None, "u_values must be provided"
#         #assert self.v_values != None, "v_values must be provided"
#     # def forward(self):
#     #     p_sorted = torch.sort(self.p_distribution, dim=0)[0]
#     #     q_sorted = torch.sort(self.q_distribution, dim=0)[0]
#     #     p_CDF = torch.arange(1, len(p_sorted) + 1).float() / len(p_sorted)
#     #     q_CDF = torch.arange(1, len(q_sorted) + 1).float() / len(q_sorted)
#     #     elementwise_CDF_distances = p_CDF - q_CDF
#     #     #x = torch.pow(torch.pow(elementwise_cumsum_distances, self.p), 1/self.p)
#     #     #y = torch.cumsum(x, dim=0)
#     #     return p_CDF#torch.abs(elementwise_CDF_distances).sum()

#     def forward(self, u_values: torch.Tensor, v_values: torch.Tensor, u_weights: Optional[torch.Tensor] = None, v_weights: Optional[torch.Tensor] = None, p: int =1):
#         u_sorter = torch.argsort(u_values)
#         v_sorter = torch.argsort(v_values)

#         all_values = torch.cat((u_values, v_values)).sort().values
#         deltas = torch.diff(all_values)

#         u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], right=True)
#         v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], right=True)

#         if u_weights is None:
#             u_cdf = u_cdf_indices.float() / u_values.size(0)
#         else:
#             u_sorted_cumweights = torch.cat((torch.tensor([0.0]), torch.cumsum(u_weights[u_sorter], 0)))
#             u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

#         if v_weights is None:
#             v_cdf = v_cdf_indices.float() / v_values.size(0)
#         else:
#             v_sorted_cumweights = torch.cat((torch.tensor([0.0]), torch.cumsum(v_weights[v_sorter], 0)))
#             v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

#         wasserstein_distance = torch.pow(torch.sum(torch.pow(torch.abs(u_cdf - v_cdf), p) * deltas), 1/p)

#         return wasserstein_distance.requires_grad_(True) # returns a tensor with gradient tracking enabled
"""
x = p - q
y = torch.cumsum(x, dim=0)
return torch.abs(y).sum()
"""

# def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
#     r"""
#     Compute the Wasserstein-1 distance between two 1D discrete distributions.

#     The Wasserstein distance, also called the Earth mover's distance or the
#     optimal transport distance, is a similarity metric between two probability
#     distributions [1]_. In the discrete case, the Wasserstein distance can be
#     understood as the cost of an optimal transport plan to convert one
#     distribution into the other. The cost is calculated as the product of the
#     amount of probability mass being moved and the distance it is being moved.
#     A brief and intuitive introduction can be found at [2]_.

#     .. versionadded:: 1.0.0

#     Parameters
#     ----------
#     u_values : 1d array_like
#         A sample from a probability distribution or the support (set of all
#         possible values) of a probability distribution. Each element is an
#         observation or possible value.

#     v_values : 1d array_like
#         A sample from or the support of a second distribution.

#     u_weights, v_weights : 1d array_like, optional
#         Weights or counts corresponding with the sample or probability masses
#         corresponding with the support values. Sum of elements must be positive
#         and finite. If unspecified, each value is assigned the same weight.

#     Returns
#     -------
#     distance : float
#         The computed distance between the distributions.

#     Notes
#     -----
#     Given two 1D probability mass functions, :math:`u` and :math:`v`, the first
#     Wasserstein distance between the distributions is:

#     .. math::

#         l_1 (u, v) = \inf_{\pi \in \Gamma (u, v)} \int_{\mathbb{R} \times
#         \mathbb{R}} |x-y| \mathrm{d} \pi (x, y)

#     where :math:`\Gamma (u, v)` is the set of (probability) distributions on
#     :math:`\mathbb{R} \times \mathbb{R}` whose marginals are :math:`u` and
#     :math:`v` on the first and second factors respectively. For a given value
#     :math:`x`, :math:`u(x)` gives the probability of :math:`u` at position
#     :math:`x`, and the same for :math:`v(x)`.

#     If :math:`U` and :math:`V` are the respective CDFs of :math:`u` and
#     :math:`v`, this distance also equals to:

#     .. math::

#         l_1(u, v) = \int_{-\infty}^{+\infty} |U-V|

#     See [3]_ for a proof of the equivalence of both definitions.

#     The input distributions can be empirical, therefore coming from samples
#     whose values are effectively inputs of the function, or they can be seen as
#     generalized functions, in which case they are weighted sums of Dirac delta
#     functions located at the specified values.

#     References
#     ----------
#     .. [1] "Wasserstein metric", https://en.wikipedia.org/wiki/Wasserstein_metric
#     .. [2] Lili Weng, "What is Wasserstein distance?", Lil'log,
#            https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance.
#     .. [3] Ramdas, Garcia, Cuturi "On Wasserstein Two Sample Testing and Related
#            Families of Nonparametric Tests" (2015). :arXiv:`1509.02237`.

#     See Also
#     --------
#     wasserstein_distance_nd: Compute the Wasserstein-1 distance between two N-D
#         discrete distributions.

#     Examples
#     --------
#     >>> from scipy.stats import wasserstein_distance
#     >>> wasserstein_distance([0, 1, 3], [5, 6, 8])
#     5.0
#     >>> wasserstein_distance([0, 1], [0, 1], [3, 1], [2, 2])
#     0.25
#     >>> wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],
#     ...                      [1.4, 0.9, 3.1, 7.2], [3.2, 3.5])
#     4.0781331438047861

#     """
#     #return _cdf_distance(1, u_values, v_values, u_weights, v_weights)



# def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
#     r"""
#     Compute, between two one-dimensional distributions :math:`u` and
#     :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
#     statistical distance that is defined as:

#     .. math::

#         l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

#     p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
#     gives the energy distance.

#     Parameters
#     ----------
#     u_values, v_values : array_like
#         Values observed in the (empirical) distribution.
#     u_weights, v_weights : array_like, optional
#         Weight for each value. If unspecified, each value is assigned the same
#         weight.
#         `u_weights` (resp. `v_weights`) must have the same length as
#         `u_values` (resp. `v_values`). If the weight sum differs from 1, it
#         must still be positive and finite so that the weights can be normalized
#         to sum to 1.

#     Returns
#     -------
#     distance : float
#         The computed distance between the distributions.

#     Notes
#     -----
#     The input distributions can be empirical, therefore coming from samples
#     whose values are effectively inputs of the function, or they can be seen as
#     generalized functions, in which case they are weighted sums of Dirac delta
#     functions located at the specified values.

#     References
#     ----------
#     .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
#            Munos "The Cramer Distance as a Solution to Biased Wasserstein
#            Gradients" (2017). :arXiv:`1705.10743`.

#     """
#     u_values, u_weights = _validate_distribution(u_values, u_weights)
#     v_values, v_weights = _validate_distribution(v_values, v_weights)

#     u_sorter = np.argsort(u_values)
#     v_sorter = np.argsort(v_values)

#     all_values = np.concatenate((u_values, v_values))
#     all_values.sort(kind='mergesort')

#     # Compute the differences between pairs of successive values of u and v.
#     deltas = np.diff(all_values)

#     # Get the respective positions of the values of u and v among the values of
#     # both distributions.
#     u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
#     v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

#     # Calculate the CDFs of u and v using their weights, if specified.
#     if u_weights is None:
#         u_cdf = u_cdf_indices / u_values.size
#     else:
#         u_sorted_cumweights = np.concatenate(([0],
#                                               np.cumsum(u_weights[u_sorter])))
#         u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

#     if v_weights is None:
#         v_cdf = v_cdf_indices / v_values.size
#     else:
#         v_sorted_cumweights = np.concatenate(([0],
#                                               np.cumsum(v_weights[v_sorter])))
#         v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

#     # Compute the value of the integral based on the CDFs.
#     # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
#     # of about 15%.
#     if p == 1:
#         return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
#     if p == 2:
#         return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
#     return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
#                                        deltas)), 1/p)

