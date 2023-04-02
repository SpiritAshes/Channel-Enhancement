#!/usr/bin/env python

import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
from numpy import *
from models.ECA import ECAAttention
from models.rep import RepVGGBlock

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])



class Channel_Enhancement(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, deploy=False):
        super(Channel_Enhancement, self).__init__()
        c1, c2, c3, c4, c5, d1 = 32, 64, 128, 256, 64, 256
        det_h = 65

        self.deploy = deploy
        self.relu = torch.nn.ReLU(inplace=True)

        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = RepVGGBlock(1, c1, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)
        self.conv2 = RepVGGBlock(c1, c1, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)
        self.conv3 = RepVGGBlock(c1, c2, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)
        self.conv4 = RepVGGBlock(c2, c2, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)
        self.conv5 = RepVGGBlock(c2, c3, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=False)
        self.conv6 = RepVGGBlock(c3, c3, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)
        self.conv7 = RepVGGBlock(c3, c4, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=False)
        self.pixel_shuffle = torch.nn.PixelShuffle(2)
        self.conv_attention = ECAAttention(3)

        self.convS_1 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnS_1 = torch.nn.BatchNorm2d(det_h)
        self.convS_2 = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnS_2 = torch.nn.BatchNorm2d(d1)

        self.convDF_1 = torch.nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1)
        self.bnDF_1 = torch.nn.BatchNorm2d(c5)
        self.convDF_2 = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnDF_2 = torch.nn.BatchNorm2d(det_h)

        self.convDD_1 = torch.nn.Conv2d(c5, d1, kernel_size=3, stride=1, padding=1)
        self.bnDD_1 = torch.nn.BatchNorm2d(d1)
        self.convDD_2 = torch.nn.Conv2d(d1, d1, kernel_size=1, stride=1, padding=0)
        self.bnDD_2 = torch.nn.BatchNorm2d(d1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.pixel_shuffle(x)
        x = self.pool(x)

        x_1 = self.bnS_1(self.convS_1(x))
        x_2 = self.bnS_2(self.convS_2(x))

        x = self.relu(self.conv_attention(x))

        Head_DF = self.relu(self.bnDF_1(self.convDF_1(x)))
        semi = self.bnDF_2(self.convDF_2(Head_DF))
        semi = semi + x_1

        Head_DD = self.relu(self.bnDD_1(self.convDD_1(x)))
        desc = self.bnDD_2(self.convDD_2(Head_DD))
        desc = desc + x_2
        dn = torch.norm(desc, p=2, dim=1)
        desc = desc.div(torch.unsqueeze(dn, 1))

        return semi, desc


class Inference(object):

  def __init__(self, weights_path, nms_dist, conf_thresh, nn_thresh,
               cuda=False):

    self.cuda = cuda
    self.nms_dist = nms_dist
    self.conf_thresh = conf_thresh
    self.nn_thresh = nn_thresh 
    self.cell = 8 
    self.border_remove = 4 
    self.net = Channel_Enhancement(deploy=True)


    if cuda:
      # Train on GPU, deploy on GPU.
      self.net.load_state_dict(torch.load(weights_path))

      # checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
      # self.net.load_state_dict(checkpoint['model_state_dict'])

      self.net = self.net.cuda()
    else:
      # Train on GPU, deploy on CPU.
      self.net.load_state_dict(torch.load(weights_path,
                               map_location=lambda storage, loc: storage))
    self.net.eval()

  def nms_fast(self, in_corners, H, W, dist_thresh):

    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

  def run(self, img):

    assert img.ndim == 2, 'Image must be grayscale.'
    assert img.dtype == np.float32, 'Image must be float32.'
    H, W = img.shape[0], img.shape[1]
    inp = img.copy()
    inp = (inp.reshape(1, H, W))
    inp = torch.from_numpy(inp)
    inp = torch.autograd.Variable(inp).view(1, 1, H, W)
    if self.cuda:
      inp = inp.cuda()
    # Forward pass of network.
    outs = self.net.forward(inp)
    semi, coarse_desc = outs[0], outs[1]
    # Convert pytorch -> numpy.
    semi = semi.data.cpu().numpy().squeeze()
    # --- Process points.
    dense = np.exp(semi) # Softmax.
    dense = dense / (np.sum(dense, axis=0)+.00001) # Should sum to 1.
    # Remove dustbin.
    nodust = dense[:-1, :, :]
    # Reshape to get full resolution heatmap.
    Hc = int(H / self.cell)
    Wc = int(W / self.cell)
    nodust = nodust.transpose(1, 2, 0)
    heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
    heatmap = np.transpose(heatmap, [0, 2, 1, 3])
    heatmap = np.reshape(heatmap, [Hc*self.cell, Wc*self.cell])
    xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
    if len(xs) == 0:
      return np.zeros((3, 0)), None, None
    pts = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist) # Apply NMS.
    inds = np.argsort(pts[2,:])
    pts = pts[:,inds[::-1]] # Sort by confidence.
    # Remove points along border.
    bord = self.border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W-bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H-bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    # --- Process descriptor.
    D = coarse_desc.shape[1]
    if pts.shape[1] == 0:
      desc = np.zeros((D, 0))
    else:
      # Interpolate into descriptor map using 2D point locations.
      samp_pts = torch.from_numpy(pts[:2, :].copy())
      samp_pts[0, :] = (samp_pts[0, :] / (float(W)/2.)) - 1.
      samp_pts[1, :] = (samp_pts[1, :] / (float(H)/2.)) - 1.
      samp_pts = samp_pts.transpose(0, 1).contiguous()
      samp_pts = samp_pts.view(1, 1, -1, 2)
      samp_pts = samp_pts.float()
      if self.cuda:
        samp_pts = samp_pts.cuda()
      desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=False)
      desc = desc.data.cpu().numpy().reshape(D, -1)
      desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return pts, desc, heatmap


class PointTracker(object):

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def nn_match_two_way(self, desc1, desc2, nn_thresh):

    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):

    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
 
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):

    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):

    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

class Image_load(object):

  def __init__(self, data_path, height, width, skip, image_suffix):
    self.listing = []
    self.sizer = [height, width]
    self.i = 0
    self.skip = skip
    self.maxlen = 1000000

    print('==> Processing Image Directory Input.')
    search = os.path.join(data_path, image_suffix)
    self.listing = glob.glob(search)
    self.listing.sort()
    self.listing = self.listing[::self.skip]
    self.maxlen = len(self.listing)
    if self.maxlen == 0:
      raise IOError('No images were found (maybe bad \'--image_suffix\' parameter?)')

  def read_frame(self):

    if self.i == self.maxlen:
      return (None, False)
    image_file = self.listing[self.i]
    gray_image = cv2.imread(image_file, 0)
    if gray_image is None:
      raise Exception('Error reading image %s' % image_file)
    # Image is resized via opencv.
    interp = cv2.INTER_AREA
    gray_image = cv2.resize(gray_image, (self.sizer[1], self.sizer[0]), interpolation=interp)
    gray_image = (gray_image.astype('float32') / 255.)
    self.i = self.i + 1
    gray_image = gray_image.astype('float32')
    return (gray_image, True)


if __name__ == '__main__':

  import yaml
  filename = 'Inference_parameters.yaml'
  with open(filename, 'r') as f:
      config = yaml.safe_load(f)

  display_scale = config['display_scale']
  H = config['H']
  W = config['W']
  weight = config['weights_path']
  nms= config['nms_dist']
  conf = config['conf_thresh']
  nn = config['nn_thresh']
  device = config['cuda']
  # This class helps load input images from different sources.
  Image_show = Image_load(config['data_path'], H, W, config['skip'], config['image_suffix'])

  print('==> Loading pre-trained network.')
  # This class runs the SuperPoint network and processes its outputs.
  fe = Inference(weights_path=config['weights_path'],
                          nms_dist=config['nms_dist'],
                          conf_thresh=config['conf_thresh'],
                          nn_thresh=config['nn_thresh'],
                          cuda=config['cuda'])
  print('==> Successfully loaded pre-trained network.')

  # This class helps merge consecutive point matches into tracks.
  tracker = PointTracker(config['max_length'], nn_thresh=fe.nn_thresh)

  # Create a window to display the demo.
  if config['display']:
    win = 'Tracker'
    cv2.namedWindow(win)
  else:
    print('Skipping visualization, will not show a GUI.')

  # Font parameters for visualizaton.
  font = cv2.FONT_HERSHEY_DUPLEX
  font_clr = (255, 255, 255)
  font_pt = (4, 12)
  font_sc = 0.4

  # Create output directory if desired.
  if config['write']:
    print('==> Will write outputs to %s' % config['write_dir'])
    if not os.path.exists(config['write_dir']):
      os.makedirs(config['write_dir'])

  print('==> Running Demo.')
  net_t_list = []
  total_t_list = []
  while True:

    start = time.time()

    # Get a new image.
    img, status = Image_show.read_frame()
    if status is False:
      break

    # Get points and descriptors.
    start1 = time.time()
    pts, desc, heatmap = fe.run(img)
    end1 = time.time()

    # Add points and descriptors to the tracker.
    tracker.update(pts, desc)

    # Get tracks for points which were match successfully across all frames.
    tracks = tracker.get_tracks(config['min_length'])

    # Primary output - Show point tracks overlayed on top of input image.
    out1 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    tracks[:, 1] /= float(fe.nn_thresh) # Normalize track scores to [0,1].
    tracker.draw_tracks(out1, tracks)
    if config['show_extra']:
      cv2.putText(out1, 'Point Tracks', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show current point detections.
    out2 = (np.dstack((img, img, img)) * 255.).astype('uint8')
    for pt in pts.T:
      pt1 = (int(round(pt[0])), int(round(pt[1])))
      cv2.circle(out2, pt1, 1, (0, 255, 0), -1, lineType=16)
    cv2.putText(out2, 'Raw Point Detections', font_pt, font, font_sc, font_clr, lineType=16)

    # Extra output -- Show the point confidence heatmap.
    if heatmap is not None:
      min_conf = 0.001
      heatmap[heatmap < min_conf] = min_conf
      heatmap = -np.log(heatmap)
      heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + .00001)
      out3 = myjet[np.round(np.clip(heatmap*10, 0, 9)).astype('int'), :]
      out3 = (out3*255).astype('uint8')
    else:
      out3 = np.zeros_like(out2)
    cv2.putText(out3, 'Raw Point Confidences', font_pt, font, font_sc, font_clr, lineType=16)

    # Resize final output.
    if config['show_extra']:
      out = np.hstack((out1, out2, out3))
      out = cv2.resize(out, (3 * display_scale * W, display_scale * H))
    else:
      out = cv2.resize(out1, (display_scale * W, display_scale * H))

    # Display visualization image to screen.
    if config['display']:
      cv2.imshow(win, out)
      key = cv2.waitKey(config['waitkey']) & 0xFF
      if key == ord('q'):
        print('Quitting, \'q\' pressed.')
        break

    # Optionally write images to disk.
    if config['write']:
      out_file = os.path.join(config['write_dir'], 'frame_%05d.png' % Image_show.i)
      print('Writing image to %s' % out_file)
      cv2.imwrite(out_file, out)

    end = time.time()

    net_t = (1./ float(end1 - start1))
    total_t = (1./ float(end - start))
    net_t_list.append(net_t)
    total_t_list.append(total_t)
    net_t_mean = mean(net_t_list)
    total_t_mean = mean(total_t_list)
    if config['show_extra']:
      print('Processed image %d (net+post_process: %.2f FPS, total: %.2f FPS, net+post_process_mean: %.2f, total_mean: %.2f).'\
            % (Image_show.i, net_t, total_t, net_t_mean, total_t_mean))

  # Close any remaining windows.
  cv2.destroyAllWindows()

  print('==> Finshed Demo.')
