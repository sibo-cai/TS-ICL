import os
import glob
from data_provider.data_loader import Dataset_tsicl, Dataset_YGTrip
from torch.utils.data import DataLoader


data_dict = {
    'tsicl': Dataset_tsicl,
    'YGTrip': Dataset_YGTrip,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size

    data_loader_list = []

    if args.data == 'tsicl':
        for f in sorted(glob.glob(f"{args.root_path}/*_{flag}.pt")):
            f = os.path.basename(f)
            terms = f.split('_')
            if len(terms) != 5:
                raise ValueError(f'Invalid file name: {f}')
            n_samples = int(terms[2])
            single_eval_pos = int(terms[3])
            data_set = Data(
                args = args,
                root_path=args.root_path,
                data_path=f,
                size=[args.seq_len, args.label_len, args.pred_len],
            )
            print(f'read in => {f}, {len(data_set)}, n_samples={n_samples}, single_eval_pos={single_eval_pos}')

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )

            data_loader_list.append([data_loader, n_samples, single_eval_pos])

    elif args.data == 'YGTrip':
        with open(os.path.join(args.root_path, args.data_path), 'r') as f:
            for i, a_line in enumerate(f):
                a_line = a_line.strip()
                if i == 0:
                    print('Header:', a_line)
                else:
                    if a_line.startswith('#'):
                        continue

                    print('read in data file =>', a_line)
                    infos = a_line.split(',')
                    city_index = int(infos[0])
                    line_index = int(infos[1])
                    data_file = infos[2]

                    data_set = Data(
                        root_path=args.root_path,
                        data_path=data_file,
                        line_index=line_index,
                        city_index=city_index,
                        flag=flag,
                        chunk_size=args.end_time_index - args.start_time_index,
                        train_n_day=args.train_n_day,
                        test_n_day=args.test_n_day,
                        day_mode=args.ygtrip_day_mode,
                    )
                    data_loader = DataLoader(
                        data_set,
                        batch_size=batch_size,
                        shuffle=shuffle_flag,
                        num_workers=args.num_workers,
                        drop_last=drop_last,
                        pin_memory=True
                    )
                    data_loader_list.append([data_loader, args.train_n_day+1, args.train_n_day])
    else:
        raise ValueError('Unsupported dataset type')

    return data_loader_list
