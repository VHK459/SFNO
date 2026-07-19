

from config import TrainConfig as cfg


clim_path = '/storage/bedartha/public/datasets/as_downloaded/weatherbench2/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr'


model = get_model()

ds_path = cfg.data_path

batch,time = prefetcher.get()


def latitude_weights(lat):
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return w


def _broadcast_weights(w: np.ndarray, ndim: int, lat_axis: int) -> np.ndarray:
    """Reshape a 1-D lat-weight vector to broadcast against an ndim array."""
    lat_axis = lat_axis % ndim
    shape = [1] * ndim
    shape[lat_axis] = w.shape[0]
    return w.reshape(shape)


def weighted_rmse_np(forecast, truth, lat, lat_axis=-2, mean_axes=None):
    """
    forecast, truth : numpy arrays; lat_axis points to the latitude dim.
    lat             : 1-D latitude values matching that axis's size.
    mean_axes       : axes to average over, e.g. (1,2,3) for
                       (channel,time,lat,lon) -> per-channel result.
    """
    w = latitude_weights(lat)
    w_b = _broadcast_weights(w, forecast.ndim, lat_axis)
    sq_err = (forecast - truth) ** 2
    return np.sqrt((sq_err * w_b).mean(axis=mean_axes))


def weighted_acc_np(forecast, truth, climatology, lat_axis=-2, mean_axes=(-2,-1)):
    w = latitude_weights(ds.latitude)
    w_b = _broadcast_weights(w.values, forecast.ndim, lat_axis)

    f_anom = forecast - climatology
    t_anom = truth - climatology
    print(t_anom.shape)
    num     = (f_anom * t_anom * w_b).mean(axis=mean_axes)
    denom_f = (f_anom ** 2 * w_b).mean(axis=mean_axes)
    denom_t = (t_anom ** 2 * w_b).mean(axis=mean_axes)

    return num / np.sqrt(denom_f * denom_t)




def rollout(model, batch, rollout_len, device, means=None, stds=None):
    """
    Autoregressive rollout -> numpy array shaped (channel, time, lat, lon),
    ready to pass directly into weighted_rmse_np / weighted_acc_np.

    batch : torch tensor (B, C, step, H, W)  -- your prefetcher's output
    means, stds : optional, shape (C,) -- if given, predictions are
                  DE-NORMALIZED back to physical units before returning
                  (RMSE/ACC should be computed in physical units, not
                  z-scored units)

    Assumes B == 1. If B > 1, call this per-sample or extend the function
    to keep a batch axis.
    """
    model.eval()
    preds = []
    current = batch[:, :, 0].to(device)          # (B, C, H, W) -- initial condition

    with torch.no_grad():
        for step in range(rollout_len):
            out = model(current)                  # (B, C, H, W)
            preds.append(out.detach().cpu().numpy())
            current = out                          # feed prediction back in autoregressively

    preds_arr = np.stack(preds, axis=0)            # (T, B, C, H, W)
    preds_arr = preds_arr[:, 0]                     # squeeze batch -> (T, C, H, W)
    preds_arr = np.moveaxis(preds_arr, 0, 1)        # -> (C, T, H, W)

    if means is not None and stds is not None:
        means_b = means.reshape(-1, 1, 1, 1)
        stds_b  = stds.reshape(-1, 1, 1, 1)
        preds_arr = preds_arr * stds_b + means_b    # back to physical units

    return preds_arr   # (C, T, H, W)

final_out = rollout(model,batch,28,DEVICE,meanss,stdss)



