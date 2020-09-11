def create_parser_calc_inception(parser): 
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--limit_category', nargs='*', help = 'List of element used for inspiration algorithm', type = str, default = ["None"])
    parser.add_argument('--size')