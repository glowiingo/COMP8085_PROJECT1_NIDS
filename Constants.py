SELECTED_FEATURES_LABEL_RFE = ['srcip', 'dstip', 'dsport', 'proto', 
                               'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 
                               'sloss', 'dloss', 'service', 'Spkts', 'swin', 'stcpb', 
                               'smeansz', 'dmeansz', 'res_bdy_len', 'Djit', 
                               'Stime', 'Ltime', 'Dintpkt', 'tcprtt', 'synack', 
                               'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst', 
                               'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

SELECTED_FEATURES_ATTACK_CAT_RFE = ['srcip', 'dsport', 'sbytes', 'dbytes', 
                                    'sttl', 'dttl', 'sloss', 'dloss', 'service', 
                                    'Spkts', 'swin', 'smeansz', 'dmeansz', 
                                    'tcprtt', 'synack', 'ct_state_ttl', 'ct_srv_dst']

SELECTED_FEATURES_LABEL_PCA = ['Ltime', 'Stime', 'is_ftp_login', 
                               'ct_ftp_cmd', 'tcprtt', 'synack', 
                               'ackdat', 'dttl', 'ct_flw_http_mthd', 
                               'swin', 'dwin', 'ct_srv_src', 'ct_srv_dst', 
                               'ct_src_dport_ltm', 'ct_dst_ltm', 'ct_src_ ltm', 
                               'stcpb', 'dtcpb', 'sttl', 'ct_dst_sport_ltm', 
                               'state', 'ct_dst_src_ltm', 'dbytes', 'dloss']

SELECTED_FEATURES_ATTACK_CAT_PCA = ['ct_srv_dst', 'ct_srv_src', 'dsport', 
                                    'ct_src_ ltm', 'ct_dst_src_ltm', 'dttl', 'ct_dst_ltm', 
                                    'ct_src_dport_ltm', 'synack', 'ct_dst_sport_ltm', 
                                    'Sload', 'sport', 'state', 'sttl', 'Dload', 
                                    'stcpb', 'service', 'res_bdy_len']

SELECTED_FEATURES_LABEL_EBFI = ['Sload', 'dstip', 'sbytes', 'srcip', 'sttl', 
                                'ct_state_ttl', 'dur', 'Dload', 'Dintpkt', 'smeansz', 
                                'dttl', 'dbytes', 'dmeansz', 'Sintpkt', 'Dpkts', 
                                'Ltime', 'Stime', 'Djit', 'sport', 'Sjit', 
                                'stcpb', 'dtcpb', 'tcprtt', 'ackdat', 'synack', 
                                'state', 'dsport', 'Spkts', 'ct_dst_sport_ltm']

SELECTED_FEATURES_ATTACK_CAT_EBFI = ['Sload', 'sbytes', 'smeansz', 'sport', 'dur', 
                                     'dstip', 'Dload', 'Dintpkt', 'ct_state_ttl', 'Ltime', 
                                     'Stime', 'sttl', 'srcip', 'dsport', 'Sintpkt', 'dbytes']

ATTACK_CAT_STR_VALUES = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoors', 'Analysis', 'Shellcode', 'Worms']