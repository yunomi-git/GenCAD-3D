# Adapted from Jung et al https://github.com/cm8908/ContrastCAD

import random
from geometry.geometry_data import GeometryLoader
from cadlib.macro import *

class Augmentations:
    def __init__(self, data_root, phase='train'):
        self.data_manager = GeometryLoader(data_root=data_root, phase=phase, with_normals=False)
        self.all_data = self.data_manager.get_all_valid_data(cad=True)

    def _get_random_cad_vec(self):
        data_id = self.all_data[random.randint(0, len(self.all_data) - 1)]
        command, args = self.data_manager.load_cad(data_id, tensor=False, pad=False)
        cad_vec = np.hstack([command[:, np.newaxis], args])
        return cad_vec

    def identity(self, command, args):
        return command, args

    def corrupt(self, command, args):
        """
        Add random 'noises' to command parameters
        """
        command, args = command.copy(), args.copy()
        seq_len = command.shape[0]
        n_commands_to_corrupt = np.random.randint(1, seq_len)
        command_to_corrupt_idx = np.random.choice(range(0, seq_len), size=n_commands_to_corrupt, replace=False)
        for idx in command_to_corrupt_idx:
            if command[idx] in [SOL_IDX, EOS_IDX]:
                continue
            n_args_to_corrupt = np.random.randint(1, N_ARGS)
            args_to_corrupt_idx = np.random.choice(range(0, N_ARGS), size=n_args_to_corrupt, replace=False)
            for j in args_to_corrupt_idx:
                if args[idx,j] == -1:
                    continue
                if command[idx] == EXT_IDX:
                    if j == 15:  # Extrude type
                        args[idx,j] = np.random.randint(0, len(EXTENT_TYPE))
                    elif j == 14:  # boolean operation
                        args[idx,j] = np.random.randint(0, len(EXTRUDE_OPERATIONS))
                elif command[idx] == ARC_IDX and j == 3:  # CCW flag {0,1}
                    args[idx,j] = np.random.randint(0, 2)
                else:
                    args[idx,j] = np.random.randint(0, ARGS_DIM)
        return command, args

    def corrupt_v2(self, command, args):
        """
        Add random 'noises' to command parameters (all the args for a `command_to_corrupt` will be corrupted)
        """
        command, args = command.copy(), args.copy()
        seq_len = command.shape[0]
        n_commands_to_corrupt = np.random.randint(1, seq_len)
        command_to_corrupt_idx = np.random.choice(range(0, seq_len), size=n_commands_to_corrupt, replace=False)
        for idx in command_to_corrupt_idx:
            if command[idx] in [SOL_IDX, EOS_IDX]:
                continue
            corruptions = np.random.randint(0, ARGS_DIM, size=(N_ARGS))  # corrupt all parameter values
            corruptions[3] = np.random.randint(0, 2)  # set range of arc CCW flag {0,1}
            corruptions[14] = np.random.randint(0, len(EXTRUDE_OPERATIONS))  # as well ext boolean operation
            corruptions[15] = np.random.randint(0, len(EXTENT_TYPE))  # ext type
            corrupted = np.ma.array(corruptions, mask=self.get_parameter_mask(command[idx]))
            args[idx] = corrupted.filled(fill_value=PAD_VAL)

        return command, args

    def noise(self, command, args, noise_level=0.1, corruption_probability=0.8):
        """
        Add random 'noises' to command parameters (all the args for a `command_to_corrupt` will be corrupted)
        """
        command, args = command.copy(), args.copy()
        seq_len = command.shape[0]
        for idx in range(seq_len):
            if np.random.rand() > corruption_probability:
                continue

            if command[idx] in [SOL_IDX, EOS_IDX]:
                continue

            translate_corruption = np.random.randint(- int(noise_level * ARGS_DIM), int(noise_level * ARGS_DIM),
                                                     size=(N_ARGS))  # corrupt all parameter values

            # add noise # This is like a box convolution filter
            new_args = args[idx] + translate_corruption
            new_args = np.clip(new_args, 1, ARGS_DIM-1)

            # Directly transfer
            constant_indices = [3, 5, 6, 7, 14, 15]
            for i in constant_indices:
                new_args[i] = args[idx, i]

            # Add padding
            new_args = np.ma.array(new_args, mask=self.get_parameter_mask(command[idx]))
            new_args = new_args.filled(fill_value=PAD_VAL)
            args[idx] = new_args

        return command, args

    def absorb(self, command, args):
        """
        Eliminate `extrude` command in order to make 2D sketch shape out of 3D cad
        """
        ext_indices = np.where(command != EXT_IDX)[0]
        command_aug = command[ext_indices]
        args_aug = args[ext_indices]
        assert command_aug.shape[0] == args_aug.shape[0]
        command_aug = np.pad(command_aug, (0, len(command) - len(command_aug)), 'constant', constant_values=EOS_IDX)
        args_aug = np.pad(args_aug, ((0, len(args) - len(args_aug)), (0, 0)), 'constant', constant_values=PAD_VAL)
        assert command_aug.shape[0] == args_aug.shape[0] and args_aug.shape[1] == N_ARGS
        return command_aug, args_aug

    def re_extrude(self, command, args):
        """
        Transform `extrude` parameters; e.g. extrude type-distance,
        Order: ..., e1, e2, boolean, type
        the extrusion type can be either `one-sided`, `symmetric`, or `two-sided` with respect to the profile’s sketch plane
        boolean operations: either creating a `new` body, or `joining`, `cutting` or `intersecting`
        """
        command, args = command.copy(), args.copy()
        ext_indices = np.where(command == EXT_IDX)[0]
        for ext_index in ext_indices:
            args[ext_index][-4:-2] = np.random.randint(0, ARGS_DIM, (2,))
            args[ext_index][-1] = np.random.randint(0, len(EXTENT_TYPE), (1,))
        return command, args


    def replace_extrusion(self, command, args):
        """
        If len(extrude) > 1, then replace one of the extrusions to another model's one
        If len == 1, then run `re_extrude`
        """
        ext_indices = np.where(command == EXT_IDX)[0]
        if len(ext_indices) > 1:
            command_aug, args_aug = self._replace(command, args, ext_indices)
        else:
            command_aug, args_aug = self.re_extrude(command, args)
        return command_aug, args_aug

    def _replace(self, command: np.ndarray, args: np.ndarray, ext_indices: np.ndarray):
        cad_vec = np.hstack([command[:,np.newaxis], args])
        ext_vec1 = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

        cad_vec2 = self._get_random_cad_vec()
        command2 = cad_vec2[:, 0]
        ext_indices2 = np.where(command2 == EXT_IDX)[0]
        ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

        n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
        old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
        new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
        for i in range(len(old_idx)):
            ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]

        sum_len = 0
        new_vec = []
        for i in range(len(ext_vec1)):
            sum_len += len(ext_vec1[i])
            if sum_len > MAX_TOTAL_LEN:
                break
            new_vec.append(ext_vec1[i])
        cad_vec = np.concatenate(new_vec, axis=0)
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        command, args = cad_vec[:,0], cad_vec[:,1:]
        return command, args

    def get_parameter_mask(self, cmd):
        mask = np.zeros((N_ARGS,))
        if cmd in [LINE_IDX, ARC_IDX, CIRCLE_IDX]:
            mask[:2] += 1
            if cmd == ARC_IDX:
                mask[2:4] += 1
            elif cmd == CIRCLE_IDX:
                mask[4] += 1
        elif cmd == EXT_IDX:
            mask[5:] += 1
        return (~mask.astype(bool)).astype(int)

    def attach(self, command, args):
        max_total_len = command.shape[0]
        command_org, args_org = command.copy(), args.copy()
        cad_vec = np.hstack([command[:,np.newaxis], args])
        ext_indices = np.where(command == EXT_IDX)[0]
        ext_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

        cad_vec2 = self._get_random_cad_vec()
        command2 = cad_vec2[:, 0]
        ext_indices2 = np.where(command2 == EXT_IDX)[0]
        ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

        # randomly select a sketch & extrusion
        rand_index = random.randint(0, len(ext_vec2) - 1)
        new_vec = ext_vec2[rand_index]

        # attach new_vec to the end of cad_vec and EOS_VEC
        cad_vec = np.concatenate(ext_vec, axis=0)
        cad_vec = np.concatenate((cad_vec, new_vec), axis=0)

        pad_len = max_total_len - cad_vec.shape[0]
        if pad_len <= 0:
            return command_org, args_org
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return cad_vec[:,0], cad_vec[:,1:]

    def translate(self, command, args):
        cad_vec = np.hstack([command[:, np.newaxis], args])
        special_token_mask = (cad_vec[:, 0] == SOL_IDX) + (cad_vec[:, 0] == EXT_IDX) + (cad_vec[:, 0] == EOS_IDX)
        min_trans = 0 - cad_vec[~special_token_mask, 1:3].min()
        max_trans = ARGS_DIM - cad_vec[:, 1:3].max()
        translate_scale = np.random.randint(min_trans, max_trans, size=(2,))
        cad_vec[:, 1:3] += translate_scale
        cad_vec[special_token_mask, 1:3] = np.ones_like(cad_vec[special_token_mask, 1:3]) * PAD_VAL
        return cad_vec[:, 0], cad_vec[:, 1:]

    def redraw(self, command, args):
        # Warning: `redraw` for CL and `redraw` for DeepCAD are differently implemented (bc. of EOS_VEC)
        max_total_len = command.shape[0]
        command_org, args_org = command.copy(), args.copy()
        cad_vec = np.hstack([command[:,np.newaxis], args])
        ext_indices = np.where(command == EXT_IDX)[0]
        ext_vec = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]
        # randomly select a setch & extrusion
        rand_index = random.randint(0, len(ext_vec) - 1)
        new_vec = ext_vec[rand_index].copy()  # e.g. [4, 0, 0, ..., 5]
        # translate the sketch
        # 1:-1 => ignore initial SOL and EXT
        # 1:3 => x,y coordinate (starting from 1 bc. 0 is command idex)
        xy_coords = new_vec[1:-1, 1:3]
        sol_mask = new_vec[1:-1, 0] == SOL_IDX  # remove parameters of intermediate SOL commands
        max_trans = ARGS_DIM - xy_coords.max()  # so that the sketch is still in the boundary
        min_trans = 0  - xy_coords[~sol_mask].min()         # so that the sketch is still in the boundary
        translate_scale = np.random.randint(min_trans, max_trans, size=(xy_coords.shape[0], 2))
        xy_coords += translate_scale
        xy_coords[sol_mask] = np.ones_like(xy_coords[sol_mask]) * PAD_VAL  # remove parameters of intermediate SOL commands
        # attach new_vec to the end of cad_vec and EOS_VEC
        cad_vec = np.concatenate(ext_vec, axis=0)
        cad_vec = np.concatenate((cad_vec, new_vec), axis=0)
        pad_len = max_total_len - cad_vec.shape[0]
        if pad_len <= 0:
            return command_org, args_org
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        assert not (cad_vec.max() >= ARGS_DIM or cad_vec.min() < -1)
        return cad_vec[:,0], cad_vec[:,1:]


    def delete(self, command, args, p=None):  # p: proportion of deletion
        if p is None:
            p = np.random.uniform()
        len_command = command[command != EOS_IDX].shape[0]
        idx_to_keep = sorted(np.random.choice(range(0, len_command), size=int(len_command * (1-p)), replace=False))
        command_aug = command[idx_to_keep].copy()
        args_aug = args[idx_to_keep].copy()
        cad_vec = np.concatenate([command_aug[:,np.newaxis], args_aug], axis=1)
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return cad_vec[:,0], cad_vec[:,1:]

    def reorder(self, command, args):
        len_command = command[command != EOS_IDX].shape[0]
        new_order = np.random.permutation(len_command)
        command_aug = command[new_order].copy()
        args_aug = args[new_order].copy()
        cad_vec = np.concatenate([command_aug[:,np.newaxis], args_aug], axis=1)
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return cad_vec[:,0], cad_vec[:,1:]

    def insert(self, command, args, p=None):
        def get_random_token():
            cad_vec2 = self._get_random_cad_vec()
            command2 = cad_vec2[:, 0]
            len_command2 = command2[command2 != EOS_IDX].shape[0]
            return cad_vec2[random.randint(0, len_command2 - 1)]
        if p is None:
            p = np.random.uniform()
        len_command = command[command != EOS_IDX].shape[0]
        cad_vec = np.hstack([command[:, np.newaxis], args])
        token_list = []
        for i in range(len_command):
            if len(token_list) < MAX_TOTAL_LEN:
                token_list.append(cad_vec[i])
            if np.random.uniform() < p and len(token_list) < MAX_TOTAL_LEN:
                random_token = get_random_token()
                token_list.append(random_token)
        cad_vec = np.stack(token_list, axis=0)
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return cad_vec[:,0], cad_vec[:,1:]

    def arc_augment(self, command, args):
        cad_vec = np.hstack([command[:, np.newaxis], args])
        if (cad_vec[:,0] == LINE_IDX).sum() == 0:
            return cad_vec[:, 0], cad_vec[:, 1:]
        target_index = random.choice(np.where(cad_vec[:, 0] == LINE_IDX)[0])
        end_x, end_y = cad_vec[target_index, 1:3]
        angle = random.randint(1, 255)
        flag = random.randint(0, 1)
        arc_token = np.array([ARC_IDX, end_x, end_y, angle, flag,] + [-1]*12)
        cad_vec[target_index] = arc_token
        return cad_vec[:,0], cad_vec[:,1:]

    # This is their original
    def arc_augment_v2(self, command, args):
        cad_vec = np.hstack([command[:, np.newaxis], args])
        num_lines = (cad_vec[:, 0] == LINE_IDX).sum()
        if num_lines == 0:
            return cad_vec[:, 0], cad_vec[:, 1:]
        num_lines_to_change = random.randint(1, num_lines)
        target_indices_to_change = np.random.choice(np.where(cad_vec[:, 0] == LINE_IDX)[0], size=num_lines_to_change,
                                                    replace=False)
        for idx in target_indices_to_change:
            end_x, end_y = cad_vec[idx, 1:3]
            angle = random.randint(1, 255)
            flag = random.randint(0, 1)
            arc_token = np.array([ARC_IDX, end_x, end_y, angle, flag, ] + [-1] * 12)
            cad_vec[idx] = arc_token
        return cad_vec[:, 0], cad_vec[:, 1:]

        # This is mine
    def arc_augment_v3(self, command, args, ratio=0.7, max_angle=np.pi * 1.0):
        cad_vec = np.hstack([command[:, np.newaxis], args])
        num_lines = (cad_vec[:,0] == LINE_IDX).sum()
        if num_lines == 0:
            return cad_vec[:, 0], cad_vec[:, 1:]
        num_lines_to_change = random.randint(1, int(np.ceil(num_lines * ratio)))
        target_indices_to_change = np.random.choice(np.where(cad_vec[:, 0] == LINE_IDX)[0], size=num_lines_to_change, replace=False)
        for idx in target_indices_to_change:
            end_x, end_y = cad_vec[idx, 1:3]
            angle = random.randint(1, int(255 * max_angle / (2 * np.pi)))
            flag = random.randint(0, 1)
            arc_token = np.array([ARC_IDX, end_x, end_y, angle, flag,] + [-1]*12)
            cad_vec[idx] = arc_token
        return cad_vec[:,0], cad_vec[:,1:]


    def rre(self, command, args):
        command_aug, args_aug = command.copy(), args.copy()
        ext_indices = np.where(command_aug == EXT_IDX)[0]
        if len(ext_indices) > 1:
            command_aug, args_aug = self._replace(command_aug, args_aug, ext_indices)
        command_aug, args_aug = self.re_extrude(command_aug, args_aug)
        command_aug, args_aug = self.arc_augment_v2(command_aug, args_aug)
        return command_aug, args_aug

    def _replace_sketch(self, command: np.ndarray, args: np.ndarray, ext_indices: np.ndarray):
        cad_vec = np.hstack([command[:,np.newaxis], args])
        ext_vec1 = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

        # data_id2 = all_data[random.randint(0, len(all_data) - 1)]
        # try:
        #     h5_path2 = os.path.join('data/cad_vec', data_id2 + ".h5")
        # except:
        #     print(data_id2)
        #     return
        # with h5py.File(h5_path2, "r") as fp:
        #     cad_vec2 = fp["vec"][:]
        cad_vec2 = self._get_random_cad_vec()
        command2 = cad_vec2[:, 0]
        ext_indices2 = np.where(command2 == EXT_IDX)[0]
        ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

        # this defines the replacement
        n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
        old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
        new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))

        # Swap the sketchs (last item is the extrude)
        for i in range(len(old_idx)):
            extrude_command = ext_vec1[old_idx[i]][-1]
            ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]
            ext_vec1[old_idx[i]][-1] = extrude_command

        # below is just padding
        sum_len = 0
        new_vec = []
        for i in range(len(ext_vec1)):
            sum_len += len(ext_vec1[i])
            if sum_len > MAX_TOTAL_LEN - 1:
                break
            new_vec.append(ext_vec1[i])
        cad_vec = np.concatenate(new_vec, axis=0)
        pad_len = MAX_TOTAL_LEN - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        command, args = cad_vec[:,0], cad_vec[:,1:]
        return command, args

    def replace_sketch(self, command, args):
        """
        If len(extrude) > 1, then replace one of the extrusions to another model's one
        If len == 1, then run `re_extrude`
        """
        ext_indices = np.where(command == EXT_IDX)[0]
        if len(ext_indices) > 1:
            command_aug, args_aug = self._replace_sketch(command, args, ext_indices)
        else:
            command_aug, args_aug = self.re_extrude(command, args)
        return command_aug, args_aug


    def dataset_augment(self, cad_vec, augment_type, aug_prob=1.0):
        if random.uniform(0, 1) < aug_prob:
            command, args = cad_vec[:, 0], cad_vec[:, 1:]
            if augment_type == 'default':
                ext_indices = np.where(command == EXT_IDX)[0]
                if len(ext_indices) > 1:
                    command, args = self._replace(command, args, ext_indices)
            elif augment_type == 're-extrude':
                command, args = self.re_extrude(command, args)
            elif augment_type == 'replace_extrusion':
                command, args = self.replace_extrusion(command, args)
            elif augment_type == 'corrupt2':
                command, args = self.corrupt_v2(command, args)
            elif augment_type == 'smallest_noise':
                command, args = self.noise(command, args, noise_level=0.01, corruption_probability=0.8)
            elif augment_type == 'small_noise':
                command, args = self.noise(command, args, noise_level=0.02, corruption_probability=0.8)
            elif augment_type == 'noise':
                command, args = self.noise(command, args, noise_level=0.07, corruption_probability=0.6)
            elif augment_type == 'redraw':
                command, args = self.redraw(command, args)
            elif augment_type == 'replace_arc':
                command, args = self.replace_extrusion(command, args)
                command, args = self.arc_augment(command, args)
            elif augment_type == 'replace_p_arc':
                command, args = self.replace_extrusion(command, args)
                if random.uniform(0, 1) > 0.5:
                    command, args = self.arc_augment(command, args)
            elif augment_type == 'replace_arc2':
                command, args = self.replace_extrusion(command, args)
                command, args = self.arc_augment_v2(command, args)
            elif augment_type == 'replace_p_arc2':
                command, args = self.replace_extrusion(command, args)
                if random.uniform(0, 1) > 0.5:
                    command, args = self.arc_augment_v2(command, args)
            elif augment_type == 'arc':
                command, args = self.arc_augment(command, args)
            elif augment_type == 'arc2':
                command, args = self.arc_augment_v2(command, args)
            elif augment_type == 'rre':
                command, args = self.rre(command, args)
            elif augment_type == 'delete':
                command, args = self.delete(command, args, p=0.2)
            elif augment_type == 'replace_sketch':
                command, args = self.replace_sketch(command, args)
            else:
                raise NotImplementedError
            cad_vec = np.hstack([command[:,np.newaxis], args])
        return cad_vec