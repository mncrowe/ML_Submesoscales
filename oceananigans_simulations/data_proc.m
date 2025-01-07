% gets oceananigans data

file_name = 'eady_turbulence_512';

disp('Loading data ...')

zeta = squeeze(ncread([file_name '_surf.nc'],'zeta'));
b = squeeze(ncread([file_name '_surf.nc'],'b'));
p = squeeze(ncread([file_name '_surf.nc'],'p'));
KE = squeeze(ncread([file_name '_surf.nc'],'KE'));

time = squeeze(ncread([file_name '_surf.nc'],'time'));
xC = squeeze(ncread([file_name '_surf.nc'],'xC'));
xF = squeeze(ncread([file_name '_surf.nc'],'xF'));
yC = squeeze(ncread([file_name '_surf.nc'],'yC'));
yF = squeeze(ncread([file_name '_surf.nc'],'yF'));

w_rms = sqrt(squeeze(ncread([file_name '_avg.nc'],'W')));
KE_avg = squeeze(ncread([file_name '_avg.nc'],'KE_avg'));
w_avg = squeeze(ncread([file_name '_avg.nc'],'w_avg'));
b_avg = squeeze(ncread([file_name '_avg.nc'],'b_avg'));

M2 = 1e-8;

B = M2*(xC-5e3) + b;

% save b, zeta frames and merge
save_frames_b = 0;
save_frames_zeta = 0;

if save_frames_b == 1; save_frames(B,'frames/frame_b','png'); end
if save_frames_zeta == 1; save_frames(zeta,'frames/frame_zeta','png'); end
if save_frames_b == 1 && save_frames_zeta == 1; system('bash conc_frames.sh'); end

% save all frames, zeta, b, KE_avg, w_rms
save_all = 1;
Nc = 256;
f = @(x) x.^0.7;
it = 1801:4801;

zeta_map = cmap2(zeta(:,:,it),0,f,Nc,0,0); [s_zeta1,s_zeta2] = F_minmax(zeta(:,:,it));
B_map = cmap2(B(:,:,it),[],f,Nc,1,0); [s_B1,s_B2] = F_minmax(B(:,:,it));
w_map = cmap2(w_avg(:,:,it),0,f,Nc,1,0); [s_w1,s_w2] = F_minmax(w_avg(:,:,it));
W_map = cmap2(w_rms(:,:,it),0,f,Nc,1,0); [s_W1,s_W2] = F_minmax(w_rms(:,:,it));

if save_all == 1
    disp('saving frames: 1/4 ...'); save_frames(zeta(:,:,it),'frames/frame_zeta','png',zeta_map,[s_zeta1,s_zeta2]);
    disp('saving frames: 2/4 ...'); save_frames(B(:,:,it),'frames/frame_B','png',B_map,[s_B1,s_B2]);
    disp('saving frames: 3/4 ...'); save_frames(w_avg(:,:,it),'frames/frame_w','png',w_map,[s_w1,s_w2]);
    disp('saving frames: 4/4 ...'); save_frames(w_rms(:,:,it),'frames/frame_W','png',W_map,[0,s_W2]);
    system('bash conc_frames_all.sh')
end


